from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cfb_quant.models.tournament import TARGETS, HOLDOUT_YEAR, _target_mask, _to_numeric_frame

DATA = Path("data/processed/cfb_pregame_features_matchup_v0_1.parquet")
WEIGHTS = Path("reports/player_matchup_blend_selected_weights_v0_1.csv")
OUTPUT = Path("reports/player_prop_holdout_predictions_2025_v0_1.csv")

def predict_artifact(artifact, frame):
    cols = artifact["feature_columns"]
    X = _to_numeric_frame(frame, cols)
    pred = artifact["model"].predict(X)
    lo, hi = artifact["prediction_bounds"]
    return np.clip(np.asarray(pred, dtype=float), lo, hi)

def main():
    frame = pd.read_parquet(DATA)
    weights = pd.read_csv(WEIGHTS).set_index("target")
    rows = []

    for target in TARGETS:
        mask = _target_mask(frame, target) & frame["season"].eq(HOLDOUT_YEAR)
        test = frame.loc[mask].copy()

        old = joblib.load(Path("models/artifacts") / f"cfb_{target}_quant_v0_1.joblib")
        new = joblib.load(Path("models/artifacts") / f"cfb_{target}_matchup_quant_v0_1.joblib")

        actual = pd.to_numeric(test[target], errors="coerce").to_numpy(dtype=float)
        pred_original = predict_artifact(old, test)
        pred_matchup = predict_artifact(new, test)

        matchup_weight = float(weights.loc[target, "selected_matchup_weight"])
        original_weight = float(weights.loc[target, "selected_original_weight"])

        pred_blended = original_weight * pred_original + matchup_weight * pred_matchup

        keep = [c for c in ["season","week","game_id","player_id","player_name","team","opponent","home_away"] if c in test.columns]
        out = test[keep].copy().reset_index(drop=True)
        out["target"] = target
        out["actual"] = actual
        out["prediction_original"] = pred_original
        out["prediction_matchup"] = pred_matchup
        out["prediction_blended"] = pred_blended
        out["blend_weight_matchup"] = matchup_weight
        out["blend_weight_original"] = original_weight
        out["residual"] = actual - pred_blended
        out["absolute_error"] = np.abs(out["residual"])
        out["model_disagreement"] = pred_matchup - pred_original
        out["absolute_model_disagreement"] = np.abs(out["model_disagreement"])

        rows.append(out)

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUTPUT, index=False)

    print("=" * 100)
    print("2025 PLAYER PROP HOLDOUT PREDICTIONS")
    print("=" * 100)
    print(f"Rows: {len(result):,}")
    print(f"Targets: {result['target'].nunique()}")
    print(f"Games: {result['game_id'].nunique():,}")
    print()
    summary = result.groupby("target").agg(
        rows=("actual","size"),
        mae=("absolute_error","mean"),
        residual_mean=("residual","mean"),
        residual_std=("residual","std"),
    ).reset_index()
    print(summary.round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
