"""Chronological Ridge + Elo residual challenger research.

Ridge is frozen; only residual correction coefficients are fitted.
No same-season training, no production modifications.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]

RIDGE = ROOT / (
    "reports/spread_total_opponent_v45/"
    "paired_game_errors.csv"
)

ELO = ROOT / (
    "reports/research/elo_spread_walkforward_v1/"
    "predictions.csv"
)

OUT = ROOT / "reports/research/elo_residual_challenger_v1"
PRED = OUT / "paired_predictions.csv"
METRICS = OUT / "season_metrics.csv"
MANIFEST = OUT / "manifest.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def main():
    require(
        not any(p.exists() for p in (PRED, METRICS, MANIFEST)),
        "Research outputs already exist. Refusing overwrite."
    )

    ridge = pd.read_csv(RIDGE, dtype={"game_id": str})
    elo = pd.read_csv(ELO, dtype={"game_id": str})

    ridge = ridge[[
        "game_id", "season",
        "actual_margin_v45", "pred_margin_v45"
    ]].rename(columns={
        "actual_margin_v45": "ridge_actual",
        "pred_margin_v45": "ridge_pred",
    })

    elo = elo[[
        "game_id", "season",
        "actual_home_margin",
        "elo_predicted_home_margin",
        "train_max_season",
    ]].rename(columns={
        "actual_home_margin": "elo_actual",
        "elo_predicted_home_margin": "elo_pred",
    })

    require(
        ridge.game_id.is_unique and elo.game_id.is_unique,
        "Duplicate source game IDs"
    )

    df = ridge.merge(
        elo,
        on=["game_id", "season"],
        how="inner",
        validate="one_to_one",
    )

    require(len(df) == 5056, "Unexpected paired game coverage")
    require(
        set(df.season.unique()) == set(range(2019, 2025)),
        "Unexpected season coverage"
    )

    numeric = [
        "ridge_actual", "ridge_pred",
        "elo_actual", "elo_pred"
    ]

    require(
        np.isfinite(df[numeric].to_numpy(dtype=float)).all(),
        "Nonfinite forecasts or outcomes"
    )

    require(
        np.allclose(
            df.ridge_actual, df.elo_actual,
            atol=0.001, rtol=0
        ),
        "Actual margin conventions disagree"
    )

    require(
        (df.train_max_season < df.season).all(),
        "Elo fitting chronology violation"
    )

    # Predict Ridge residual from the disagreement
    # between Elo and Ridge. The target is used only in fitting.
    df["disagreement"] = df.elo_pred - df.ridge_pred
    df["ridge_residual"] = df.ridge_actual - df.ridge_pred

    predictions = []
    metrics = []

    for year in range(2021, 2025):
        train = df.loc[df.season < year].copy()
        test = df.loc[df.season == year].copy()

        require(
            not train.empty and not test.empty,
            f"Empty split for {year}"
        )

        require(
            train.season.max() < test.season.min(),
            "Training chronology violation"
        )

        model = LinearRegression()

        model.fit(
            train[["disagreement"]],
            train["ridge_residual"]
        )

        correction = model.predict(
            test[["disagreement"]]
        )

        challenger = test.ridge_pred.to_numpy() + correction
        actual = test.ridge_actual.to_numpy()
        champion = test.ridge_pred.to_numpy()

        champion_mae = mae(actual, champion)
        challenger_mae = mae(actual, challenger)

        rows = test[[
            "game_id", "season", "ridge_pred",
            "elo_pred", "ridge_actual", "disagreement"
        ]].copy()

        rows["challenger_pred"] = challenger
        rows["ridge_abs_error"] = np.abs(actual - champion)
        rows["challenger_abs_error"] = np.abs(actual - challenger)
        rows["train_max_season"] = int(train.season.max())

        predictions.append(rows)

        metrics.append({
            "season": year,
            "train_games": len(train),
            "test_games": len(test),
            "ridge_mae": champion_mae,
            "challenger_mae": challenger_mae,
            "improvement": champion_mae - challenger_mae,
            "correction_intercept": float(model.intercept_),
            "disagreement_coefficient": float(model.coef_[0]),
        })

    result = pd.concat(predictions, ignore_index=True)
    summary = pd.DataFrame(metrics)

    require(
        result.game_id.is_unique,
        "Duplicate output predictions"
    )
    require(
        (result.train_max_season < result.season).all(),
        "Invalid training cutoff"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(PRED, index=False)
    summary.to_csv(METRICS, index=False)

    manifest = {
        "schema": "elo_residual_challenger_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True
        ).strip(),
        "ridge_source_sha256": sha256(RIDGE),
        "elo_source_sha256": sha256(ELO),
        "predictions_sha256": sha256(PRED),
        "metrics_sha256": sha256(METRICS),
        "training_policy": "strictly_earlier_seasons",
        "evaluation_seasons": [2021, 2022, 2023, 2024],
        "model": "linear_residual_correction_from_elo_ridge_disagreement",
        "historical_point_in_time_certified": False,
        "ridge_oof_training_provenance_certified": False,
        "production_approved": False,
        "research_only": True,
    }

    with MANIFEST.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("\n=== RIDGE + ELO RESIDUAL CHALLENGER ===")
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    print("\n=== ALL EVALUATION GAMES ===")
    print("Games:", len(result))
    print("Ridge MAE:", round(result.ridge_abs_error.mean(), 4))
    print(
        "Challenger MAE:",
        round(result.challenger_abs_error.mean(), 4)
    )
    print(
        "Improvement:",
        round(
            result.ridge_abs_error.mean()
            - result.challenger_abs_error.mean(),
            4
        )
    )

    holdout = result[result.season == 2024]

    print("\n=== SEPARATELY REPORTED 2024 EVALUATION ===")
    print("Games:", len(holdout))
    print("Ridge MAE:", round(holdout.ridge_abs_error.mean(), 4))
    print(
        "Challenger MAE:",
        round(holdout.challenger_abs_error.mean(), 4)
    )
    print(
        "Improvement:",
        round(
            holdout.ridge_abs_error.mean()
            - holdout.challenger_abs_error.mean(),
            4
        )
    )

    print("\nProduction modified: NO")
    print("Historical PIT certified: NO")
    print("Output:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
