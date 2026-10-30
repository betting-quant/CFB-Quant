"""Season-blocked Elo-only spread research.

Fits a one-feature linear regression using historical pregame
Elo rating difference. Each test season is strictly held out.

Experimental retrospective data only.
No production changes or market inputs.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / (
    "reports/research/historical_elo_v1/"
    "pregame_elo_features.csv"
)

OUT = ROOT / "reports/research/elo_spread_walkforward_v1"
PREDICTIONS = OUT / "predictions.csv"
METRICS = OUT / "season_metrics.csv"
MANIFEST = OUT / "manifest.json"

FIRST_TEST_YEAR = 2019
LAST_TEST_YEAR = 2025


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    require(SOURCE.is_file(), "Historical Elo file missing")

    require(
        not any(p.exists() for p in [
            PREDICTIONS, METRICS, MANIFEST
        ]),
        "Existing walk-forward outputs detected"
    )

    df = pd.read_csv(SOURCE, dtype={"game_id": str})

    columns = [
        "game_id", "season", "elo_rating_difference",
        "actual_home_margin", "kickoff_utc"
    ]

    require(
        set(columns).issubset(df.columns),
        "Missing required Elo data columns"
    )

    require(
        not df.game_id.duplicated().any(),
        "Duplicate games in Elo source"
    )

    require(
        df[columns].notna().all().all(),
        "Missing Elo model inputs or targets"
    )

    for col in [
        "elo_rating_difference", "actual_home_margin"
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    require(
        np.isfinite(
            df[[
                "elo_rating_difference",
                "actual_home_margin"
            ]].to_numpy()
        ).all(),
        "Nonfinite Elo inputs or targets"
    )

    df["season"] = pd.to_numeric(
        df["season"], errors="raise"
    ).astype(int)

    df["kickoff_utc"] = pd.to_datetime(
        df["kickoff_utc"], utc=True, errors="raise"
    )

    predictions = []
    metrics = []

    for year in range(
        FIRST_TEST_YEAR, LAST_TEST_YEAR + 1
    ):
        train = df.loc[df.season < year].copy()
        test = df.loc[df.season == year].copy()

        require(
            not train.empty and not test.empty,
            f"Missing train/test data for {year}"
        )

        require(
            train.season.max() < year,
            f"Training season leakage for {year}"
        )

        require(
            train.kickoff_utc.max()
            < test.kickoff_utc.min(),
            f"Chronological train/test overlap for {year}"
        )

        X_train = train[["elo_rating_difference"]]
        y_train = train["actual_home_margin"]

        X_test = test[["elo_rating_difference"]]
        y_test = test["actual_home_margin"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        require(
            np.isfinite(y_pred).all(),
            f"Nonfinite predictions for {year}"
        )

        result = pd.DataFrame({
            "game_id": test.game_id.to_numpy(),
            "season": year,
            "kickoff_utc": test.kickoff_utc.dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ).to_numpy(),
            "elo_rating_difference":
                test.elo_rating_difference.to_numpy(),
            "actual_home_margin": y_test.to_numpy(),
            "elo_predicted_home_margin": y_pred,
            "absolute_error": np.abs(
                y_test.to_numpy() - y_pred
            ),
            "train_max_season": train.season.max(),
            "train_games": len(train),
        })

        predictions.append(result)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        metrics.append({
            "season": year,
            "train_games": len(train),
            "test_games": len(test),
            "mae": mae,
            "rmse": rmse,
            "intercept": float(model.intercept_),
            "elo_coefficient": float(model.coef_[0]),
        })

    pred = pd.concat(
        predictions, ignore_index=True
    ).sort_values(
        ["season", "kickoff_utc", "game_id"]
    ).reset_index(drop=True)

    summary = pd.DataFrame(metrics)

    require(
        pred.game_id.is_unique,
        "Duplicate game-level predictions"
    )

    require(
        (pred.train_max_season < pred.season).all(),
        "Walk-forward training leakage detected"
    )

    overall_mae = pred.absolute_error.mean()
    overall_rmse = np.sqrt(
        np.mean(
            (
                pred.actual_home_margin
                - pred.elo_predicted_home_margin
            ) ** 2
        )
    )

    OUT.mkdir(parents=True, exist_ok=True)

    pred.to_csv(PREDICTIONS, index=False)
    summary.to_csv(METRICS, index=False)

    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    manifest = {
        "schema": "elo_spread_walkforward_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_revision": git_revision,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "predictions_sha256": sha256(PREDICTIONS),
        "metrics_sha256": sha256(METRICS),
        "training_policy": "strictly_earlier_seasons",
        "model": "linear_regression_elo_difference_only",
        "target": "actual_home_margin",
        "margin_convention": "home_minus_away",
        "test_seasons": list(
            range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)
        ),
        "test_games": len(pred),
        "overall_mae": float(overall_mae),
        "overall_rmse": float(overall_rmse),
        "historical_point_in_time_certified": False,
        "eligibility": (
            "source historical matchup universe; "
            "FBS-only equivalence not yet certified"
        ),
        "production_approved": False,
        "research_only": True,
    }

    with MANIFEST.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("\n=== ELO SPREAD WALK-FORWARD RESULTS ===")
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    print("\nOVERALL")
    print("Test games:", len(pred))
    print("MAE:", round(overall_mae, 4))
    print("RMSE:", round(overall_rmse, 4))
    print("Chronological training: PASS")
    print("Historical PIT certified: NO")
    print("Production modified: NO")
    print("Output:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
