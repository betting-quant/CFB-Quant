from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

INPUT = Path("reports/player_probability_calibration_2025_v0_1.csv")
OUTPUT = Path("reports/player_probability_calibrator_validation_2025_v0_1.csv")
ARTIFACT = Path("models/artifacts/cfb_player_probability_calibrators_v0_1.joblib")

def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))

def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    score = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not mask.any():
            continue
        score += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(score)

def main():
    df = pd.read_csv(INPUT)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["target","week","predicted_over_probability","actual_over"]).copy()

    calibrators = {}
    results = []

    print("=" * 115)
    print("2025 TEMPORAL PROBABILITY CALIBRATION VALIDATION")
    print("=" * 115)

    for target, group in df.groupby("target", sort=False):
        weeks = sorted(group["week"].unique())
        split_index = max(1, int(np.floor(len(weeks) * 0.70)))
        split_index = min(split_index, len(weeks) - 1)

        train_weeks = weeks[:split_index]
        test_weeks = weeks[split_index:]

        train = group[group["week"].isin(train_weeks)].copy()
        test = group[group["week"].isin(test_weeks)].copy()

        iso = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            increasing=True,
            out_of_bounds="clip",
        )

        iso.fit(
            train["predicted_over_probability"].to_numpy(dtype=float),
            train["actual_over"].to_numpy(dtype=float),
        )

        raw_p = test["predicted_over_probability"].to_numpy(dtype=float)
        y = test["actual_over"].to_numpy(dtype=float)
        cal_p = iso.predict(raw_p)

        raw_brier = brier(y, raw_p)
        calibrated_brier = brier(y, cal_p)
        raw_ece = ece(y, raw_p)
        calibrated_ece = ece(y, cal_p)

        results.append({
            "target": target,
            "train_weeks": ",".join(str(int(x)) for x in train_weeks),
            "validation_weeks": ",".join(str(int(x)) for x in test_weeks),
            "train_rows": len(train),
            "validation_rows": len(test),
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "brier_improvement": raw_brier - calibrated_brier,
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
            "ece_improvement": raw_ece - calibrated_ece,
        })

        # Refit final calibrator on all 2025 rows only after validation.
        final_iso = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            increasing=True,
            out_of_bounds="clip",
        )
        final_iso.fit(
            group["predicted_over_probability"].to_numpy(dtype=float),
            group["actual_over"].to_numpy(dtype=float),
        )
        calibrators[target] = final_iso

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT, index=False)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrators, ARTIFACT)

    print(result.round(5).to_string(index=False))
    print()
    print("Targets improved on Brier:", int((result["brier_improvement"] > 0).sum()), "/", len(result))
    print("Targets improved on ECE:", int((result["ece_improvement"] > 0).sum()), "/", len(result))
    print()
    print("Saved:", OUTPUT)
    print("Saved:", ARTIFACT)

if __name__ == "__main__":
    main()
