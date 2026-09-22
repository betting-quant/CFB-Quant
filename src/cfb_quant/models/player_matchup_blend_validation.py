from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import cfb_quant.models.tournament as base


ORIGINAL_DATA = Path(
    "data/processed/cfb_pregame_features_v0_1.parquet"
)

ORIGINAL_MANIFEST = Path(
    "reports/cfb_feature_manifest_v0_1.json"
)

MATCHUP_DATA = Path(
    "data/processed/cfb_pregame_features_matchup_v0_1.parquet"
)

MATCHUP_MANIFEST = Path(
    "reports/cfb_feature_manifest_matchup_v0_1.json"
)

ORIGINAL_SUMMARY = Path(
    "reports/model_tournament_summary_v0_1.csv"
)

MATCHUP_SUMMARY = Path(
    "reports/model_tournament_matchup_summary_v0_1.csv"
)

OUTPUT = Path(
    "reports/player_matchup_blend_validation_v0_1.csv"
)

SELECTED_OUTPUT = Path(
    "reports/player_matchup_blend_selected_weights_v0_1.csv"
)

WEIGHTS = np.arange(
    0.0,
    1.0001,
    0.05,
)


def choose_winners(path):
    summary = pd.read_csv(path)

    winners = {}

    for target in base.TARGETS:
        rows = summary.loc[
            summary["target"].eq(target)
        ].sort_values(
            ["mean_mae", "mean_rmse"]
        )

        winners[target] = str(
            rows.iloc[0]["model"]
        )

    return winners


def metrics(actual, pred):
    error = pred - actual

    return {
        "mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "bias": float(
            np.mean(error)
        ),
        "corr": float(
            np.corrcoef(
                actual,
                pred,
            )[0, 1]
        ),
    }


def main():

    original = pd.read_parquet(
        ORIGINAL_DATA
    )

    matchup = pd.read_parquet(
        MATCHUP_DATA
    )

    original_manifest = json.loads(
        ORIGINAL_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    matchup_manifest = json.loads(
        MATCHUP_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    original_features = list(
        original_manifest[
            "feature_columns"
        ]
    )

    matchup_features = list(
        matchup_manifest[
            "feature_columns"
        ]
    )

    original_winners = choose_winners(
        ORIGINAL_SUMMARY
    )

    matchup_winners = choose_winners(
        MATCHUP_SUMMARY
    )

    validation_years = [
        2021,
        2022,
        2023,
        2024,
    ]

    all_rows = []
    selected = []

    print("=" * 110)
    print("CFB PLAYER BLEND — VALIDATION-ONLY WEIGHT SELECTION")
    print("=" * 110)

    for target in base.TARGETS:

        actual_all = []
        original_pred_all = []
        matchup_pred_all = []

        print()
        print(target)
        print("-" * 110)

        for year in validation_years:

            original_eligible = (
                base._target_mask(
                    original,
                    target,
                )
            )

            matchup_eligible = (
                base._target_mask(
                    matchup,
                    target,
                )
            )

            original_train = original.loc[
                original_eligible
                & original[
                    "season"
                ].lt(year)
            ]

            original_val = original.loc[
                original_eligible
                & original[
                    "season"
                ].eq(year)
            ]

            matchup_train = matchup.loc[
                matchup_eligible
                & matchup[
                    "season"
                ].lt(year)
            ]

            matchup_val = matchup.loc[
                matchup_eligible
                & matchup[
                    "season"
                ].eq(year)
            ]

            if (
                len(original_val)
                != len(matchup_val)
            ):
                raise RuntimeError(
                    f"{target} {year}: "
                    "validation row mismatch"
                )

            old_model_name = (
                original_winners[target]
            )

            new_model_name = (
                matchup_winners[target]
            )

            X_old_train = (
                base._to_numeric_frame(
                    original_train,
                    original_features,
                )
            )

            X_old_val = (
                base._to_numeric_frame(
                    original_val,
                    original_features,
                )
            )

            y_old_train = pd.to_numeric(
                original_train[target],
                errors="coerce",
            )

            _, old_pred = (
                base._fit_and_predict(
                    old_model_name,
                    target,
                    X_old_train,
                    y_old_train,
                    X_old_val,
                )
            )

            X_new_train = (
                base._to_numeric_frame(
                    matchup_train,
                    matchup_features,
                )
            )

            X_new_val = (
                base._to_numeric_frame(
                    matchup_val,
                    matchup_features,
                )
            )

            y_new_train = pd.to_numeric(
                matchup_train[target],
                errors="coerce",
            )

            _, new_pred = (
                base._fit_and_predict(
                    new_model_name,
                    target,
                    X_new_train,
                    y_new_train,
                    X_new_val,
                )
            )

            actual = pd.to_numeric(
                original_val[target],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            actual_all.append(
                actual
            )

            original_pred_all.append(
                np.asarray(
                    old_pred,
                    dtype=float,
                )
            )

            matchup_pred_all.append(
                np.asarray(
                    new_pred,
                    dtype=float,
                )
            )

            print(
                f"{year}: "
                f"{len(actual):,} rows"
            )

        actual = np.concatenate(
            actual_all
        )

        old_pred = np.concatenate(
            original_pred_all
        )

        new_pred = np.concatenate(
            matchup_pred_all
        )

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
                "target": target,
                "matchup_weight":
                    float(
                        matchup_weight
                    ),
                "original_weight":
                    float(
                        original_weight
                    ),
                "rows":
                    int(len(actual)),
                "original_model":
                    original_winners[
                        target
                    ],
                "matchup_model":
                    matchup_winners[
                        target
                    ],
                **result,
            }

            all_rows.append(row)
            target_rows.append(row)

        table = pd.DataFrame(
            target_rows
        )

        best = table.sort_values(
            [
                "mae",
                "rmse",
            ]
        ).iloc[0]

        original_row = table.loc[
            table[
                "matchup_weight"
            ].eq(0.0)
        ].iloc[0]

        matchup_row = table.loc[
            table[
                "matchup_weight"
            ].eq(1.0)
        ].iloc[0]

        selected.append(
            {
                "target": target,
                "selected_matchup_weight":
                    float(
                        best[
                            "matchup_weight"
                        ]
                    ),
                "selected_original_weight":
                    float(
                        best[
                            "original_weight"
                        ]
                    ),
                "validation_mae":
                    float(
                        best["mae"]
                    ),
                "original_validation_mae":
                    float(
                        original_row[
                            "mae"
                        ]
                    ),
                "matchup_validation_mae":
                    float(
                        matchup_row[
                            "mae"
                        ]
                    ),
                "validation_mae_gain_vs_original":
                    float(
                        original_row[
                            "mae"
                        ]
                        - best["mae"]
                    ),
                "validation_rmse":
                    float(
                        best["rmse"]
                    ),
                "validation_corr":
                    float(
                        best["corr"]
                    ),
                "original_model":
                    original_winners[
                        target
                    ],
                "matchup_model":
                    matchup_winners[
                        target
                    ],
            }
        )

        print()
        print(
            "SELECTED:",
            f"matchup={best['matchup_weight']:.2f}",
            f"original={best['original_weight']:.2f}",
            f"MAE={best['mae']:.5f}",
            f"original MAE={original_row['mae']:.5f}",
            f"matchup MAE={matchup_row['mae']:.5f}",
        )

    result_df = pd.DataFrame(
        all_rows
    )

    selected_df = pd.DataFrame(
        selected
    )

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    selected_df.to_csv(
        SELECTED_OUTPUT,
        index=False,
    )

    print()
    print("=" * 110)
    print("VALIDATION-SELECTED BLEND WEIGHTS")
    print("=" * 110)

    print(
        selected_df.round(5)
        .to_string(
            index=False
        )
    )

    print()
    print("Saved:", OUTPUT)
    print("Saved:", SELECTED_OUTPUT)


if __name__ == "__main__":
    main()
