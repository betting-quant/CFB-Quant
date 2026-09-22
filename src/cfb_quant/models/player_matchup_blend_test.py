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
    "player_matchup_blend_holdout_2025_v0_1.csv"
)

WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
]


def predict_artifact(
    artifact,
    frame,
):
    cols = artifact[
        "feature_columns"
    ]

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


def metrics(
    actual,
    pred,
):
    error = (
        pred
        - actual
    )

    abs_error = np.abs(
        error
    )

    mae = float(
        abs_error.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    bias = float(
        error.mean()
    )

    if (
        len(actual) >= 2
        and np.std(actual) > 0
        and np.std(pred) > 0
    ):
        corr = float(
            np.corrcoef(
                actual,
                pred,
            )[0, 1]
        )
    else:
        corr = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
    }


def main():

    frame = pd.read_parquet(
        DATA
    )

    rows = []

    print("=" * 115)
    print("2025 HOLDOUT BLEND TEST")
    print("=" * 115)

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

        actual = pd.to_numeric(
            test[target],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        old = joblib.load(
            Path(
                "models/artifacts/"
                f"cfb_{target}_"
                "quant_v0_1.joblib"
            )
        )

        new = joblib.load(
            Path(
                "models/artifacts/"
                f"cfb_{target}_"
                "matchup_quant_v0_1.joblib"
            )
        )

        old_pred = predict_artifact(
            old,
            test,
        )

        new_pred = predict_artifact(
            new,
            test,
        )

        print()
        print(target)
        print("-" * 115)

        target_rows = []

        for matchup_weight in WEIGHTS:

            original_weight = (
                1.0
                - matchup_weight
            )

            blend = (
                original_weight
                * old_pred
                + matchup_weight
                * new_pred
            )

            result = metrics(
                actual,
                blend,
            )

            row = {
                "target":
                    target,
                "matchup_weight":
                    matchup_weight,
                "original_weight":
                    original_weight,
                **result,
            }

            rows.append(
                row
            )

            target_rows.append(
                row
            )

        target_df = pd.DataFrame(
            target_rows
        )

        best = target_df.sort_values(
            [
                "mae",
                "rmse",
                "bias",
            ],
            key=lambda s:
                s.abs()
                if s.name == "bias"
                else s,
        ).iloc[0]

        print(
            target_df[
                [
                    "matchup_weight",
                    "original_weight",
                    "mae",
                    "rmse",
                    "bias",
                    "corr",
                ]
            ]
            .round(5)
            .to_string(
                index=False
            )
        )

        print()
        print(
            "BEST:",
            f"matchup={best['matchup_weight']:.2f}",
            f"original={best['original_weight']:.2f}",
            f"MAE={best['mae']:.5f}",
            f"RMSE={best['rmse']:.5f}",
        )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 115)
    print("BEST BLEND BY TARGET")
    print("=" * 115)

    best_rows = []

    for target, group in out.groupby(
        "target",
        sort=False,
    ):

        best = group.sort_values(
            [
                "mae",
                "rmse",
            ]
        ).iloc[0]

        best_rows.append(
            best
        )

    best_df = pd.DataFrame(
        best_rows
    )

    print(
        best_df[
            [
                "target",
                "matchup_weight",
                "original_weight",
                "mae",
                "rmse",
                "bias",
                "corr",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
