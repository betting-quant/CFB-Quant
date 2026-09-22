from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cfb_quant.models.tournament import (
    TARGETS,
    HOLDOUT_YEAR,
    _target_mask,
    _to_numeric_frame,
)


DATA = Path(
    "data/processed/"
    "cfb_pregame_features_matchup_v0_1.parquet"
)

OUTPUT = Path(
    "reports/"
    "player_matchup_paired_holdout_2025_v0_1.csv"
)


def predict_artifact(
    artifact,
    frame,
):
    cols = artifact["feature_columns"]

    X = _to_numeric_frame(
        frame,
        cols,
    )

    pred = artifact[
        "model"
    ].predict(X)

    lo, hi = artifact[
        "prediction_bounds"
    ]

    return np.clip(
        np.asarray(
            pred,
            dtype=float,
        ),
        lo,
        hi,
    )


def main():

    frame = pd.read_parquet(
        DATA
    )

    results = []

    print("=" * 105)
    print("PAIRED 2025 HOLDOUT — ORIGINAL vs MATCHUP")
    print("=" * 105)

    for target in TARGETS:

        mask = (
            _target_mask(
                frame,
                target,
            )
            & frame[
                "season"
            ].eq(
                HOLDOUT_YEAR
            )
        )

        test = frame.loc[
            mask
        ].copy()

        old_path = Path(
            "models/artifacts/"
            f"cfb_{target}_quant_v0_1.joblib"
        )

        new_path = Path(
            "models/artifacts/"
            f"cfb_{target}_matchup_quant_v0_1.joblib"
        )

        old = joblib.load(
            old_path
        )

        new = joblib.load(
            new_path
        )

        actual = pd.to_numeric(
            test[target],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        old_pred = predict_artifact(
            old,
            test,
        )

        new_pred = predict_artifact(
            new,
            test,
        )

        old_error = np.abs(
            old_pred - actual
        )

        new_error = np.abs(
            new_pred - actual
        )

        delta = (
            new_error
            - old_error
        )

        improved = (
            new_error
            < old_error
        )

        worsened = (
            new_error
            > old_error
        )

        tied = (
            new_error
            == old_error
        )

        rng = np.random.default_rng(
            42
        )

        bootstrap = []

        n = len(delta)

        for _ in range(5000):
            idx = rng.integers(
                0,
                n,
                n,
            )

            bootstrap.append(
                float(
                    np.mean(
                        delta[idx]
                    )
                )
            )

        bootstrap = np.asarray(
            bootstrap
        )

        ci_low = float(
            np.quantile(
                bootstrap,
                0.025,
            )
        )

        ci_high = float(
            np.quantile(
                bootstrap,
                0.975,
            )
        )

        mean_delta = float(
            np.mean(delta)
        )

        median_delta = float(
            np.median(delta)
        )

        row = {
            "target": target,
            "rows": n,
            "old_mae":
                float(
                    old_error.mean()
                ),
            "new_mae":
                float(
                    new_error.mean()
                ),
            "mean_absolute_error_delta":
                mean_delta,
            "median_absolute_error_delta":
                median_delta,
            "improved_rows":
                int(
                    improved.sum()
                ),
            "worsened_rows":
                int(
                    worsened.sum()
                ),
            "tied_rows":
                int(
                    tied.sum()
                ),
            "improved_row_pct":
                float(
                    improved.mean()
                    * 100
                ),
            "bootstrap_delta_ci_low":
                ci_low,
            "bootstrap_delta_ci_high":
                ci_high,
            "bootstrap_supports_improvement":
                bool(
                    ci_high < 0
                ),
        }

        results.append(
            row
        )

        print()
        print(target)
        print("-" * 105)

        print(
            f"Rows: {n:,}"
        )

        print(
            f"Old MAE: "
            f"{row['old_mae']:.5f}"
        )

        print(
            f"New MAE: "
            f"{row['new_mae']:.5f}"
        )

        print(
            f"Mean error delta: "
            f"{mean_delta:+.5f}"
        )

        print(
            f"Rows improved: "
            f"{row['improved_row_pct']:.2f}%"
        )

        print(
            "Bootstrap 95% CI: "
            f"[{ci_low:+.5f}, "
            f"{ci_high:+.5f}]"
        )

        print(
            "Statistical support for "
            "improvement:",
            row[
                "bootstrap_supports_improvement"
            ],
        )

    out = pd.DataFrame(
        results
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 105)
    print("SUMMARY")
    print("=" * 105)

    print(
        out[
            [
                "target",
                "old_mae",
                "new_mae",
                "mean_absolute_error_delta",
                "improved_row_pct",
                "bootstrap_delta_ci_low",
                "bootstrap_delta_ci_high",
                "bootstrap_supports_improvement",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Negative error delta = "
        "matchup model better."
    )

    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
