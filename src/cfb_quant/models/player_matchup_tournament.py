from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import cfb_quant.models.tournament as base


DEFAULT_DATA = (
    "data/processed/"
    "cfb_pregame_features_matchup_v0_1.parquet"
)

DEFAULT_MANIFEST = (
    "reports/"
    "cfb_feature_manifest_matchup_v0_1.json"
)

REPORT_DIR = Path("reports")
ARTIFACT_DIR = Path("models/artifacts")


def fit_matchup_holdout(
    frame: pd.DataFrame,
    feature_columns: list[str],
    winners: dict[str, str],
    summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for target in base.TARGETS:

        eligible = base._target_mask(
            frame,
            target,
        )

        train_mask = (
            eligible
            & frame["season"].between(
                base.MIN_TRAIN_SEASON,
                base.HOLDOUT_YEAR - 1,
            )
        )

        holdout_mask = (
            eligible
            & frame["season"].eq(
                base.HOLDOUT_YEAR
            )
        )

        train = frame.loc[
            train_mask
        ]

        holdout = frame.loc[
            holdout_mask
        ]

        if train.empty or holdout.empty:
            raise RuntimeError(
                f"{target}: empty split"
            )

        if int(
            train["season"].max()
        ) != base.HOLDOUT_YEAR - 1:
            raise RuntimeError(
                f"{target}: training cutoff wrong"
            )

        if base.HOLDOUT_YEAR in set(
            train["season"]
        ):
            raise RuntimeError(
                "2025 holdout leakage"
            )

        model_name = winners[target]

        X_train = base._to_numeric_frame(
            train,
            feature_columns,
        )

        y_train = pd.to_numeric(
            train[target],
            errors="coerce",
        )

        X_holdout = base._to_numeric_frame(
            holdout,
            feature_columns,
        )

        y_holdout = pd.to_numeric(
            holdout[target],
            errors="coerce",
        ).to_numpy(dtype=float)

        model, pred = base._fit_and_predict(
            model_name,
            target,
            X_train,
            y_train,
            X_holdout,
        )

        metric = base._metrics(
            y_holdout,
            pred,
        )

        winner_validation = summary.loc[
            (
                summary["target"].eq(
                    target
                )
            )
            & (
                summary["model"].eq(
                    model_name
                )
            )
        ].iloc[0]

        artifact = {
            "artifact_version": "v0.1",
            "artifact_type":
                "cfb_player_matchup_model",
            "target": target,
            "model_name": model_name,
            "model": model,
            "feature_columns":
                feature_columns,
            "feature_count":
                len(feature_columns),
            "training_rows":
                int(len(train)),
            "training_seasons":
                sorted(
                    int(x)
                    for x
                    in train[
                        "season"
                    ].unique()
                ),
            "max_train_season":
                int(
                    train[
                        "season"
                    ].max()
                ),
            "eligibility_family":
                base.TARGET_FAMILY[
                    target
                ],
            "prediction_bounds":
                base.PREDICTION_BOUNDS[
                    target
                ],
            "validation_mean_mae":
                float(
                    winner_validation[
                        "mean_mae"
                    ]
                ),
            "validation_mean_rmse":
                float(
                    winner_validation[
                        "mean_rmse"
                    ]
                ),
            "holdout_year":
                base.HOLDOUT_YEAR,
            "holdout_rows":
                int(len(holdout)),
            "holdout_mae":
                metric["mae"],
            "holdout_rmse":
                metric["rmse"],
            "holdout_bias":
                metric["bias"],
            "holdout_corr":
                metric["corr"],
            "holdout_used_for_model_selection":
                False,
            "matchup_features": [
                c
                for c
                in feature_columns
                if c.startswith(
                    "matchup_proj_"
                )
            ],
        }

        artifact_path = (
            ARTIFACT_DIR
            / (
                f"cfb_{target}_"
                "matchup_quant_v0_1.joblib"
            )
        )

        joblib.dump(
            artifact,
            artifact_path,
        )

        rows.append(
            {
                "target": target,
                "winner": model_name,
                "train_rows":
                    int(len(train)),
                "holdout_rows":
                    int(len(holdout)),
                **metric,
                "artifact":
                    str(artifact_path),
            }
        )

        print(
            f"HOLDOUT "
            f"{target:18s} | "
            f"{model_name:22s} | "
            f"MAE {metric['mae']:.4f} | "
            f"RMSE {metric['rmse']:.4f}"
        )

    return pd.DataFrame(
        rows
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    args = parser.parse_args()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "CFB MATCHUP-ENHANCED "
        "PLAYER TOURNAMENT v0.1"
    )
    print("=" * 90)

    frame = pd.read_parquet(
        args.data
    )

    manifest = json.loads(
        Path(
            args.manifest
        ).read_text(
            encoding="utf-8"
        )
    )

    feature_columns = list(
        manifest[
            "feature_columns"
        ]
    )

    missing = sorted(
        set(feature_columns)
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing features: "
            f"{missing[:20]}"
        )

    matchup_features = [
        c
        for c in feature_columns
        if c.startswith(
            "matchup_proj_"
        )
    ]

    if len(
        matchup_features
    ) != 7:
        raise RuntimeError(
            "Expected exactly 7 "
            "matchup features; "
            f"found {len(matchup_features)}"
        )

    print(
        f"Rows loaded: "
        f"{len(frame):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns):,}"
    )

    print(
        f"Matchup features: "
        f"{len(matchup_features)}"
    )

    print(
        f"Seasons: "
        f"{int(frame['season'].min())}-"
        f"{int(frame['season'].max())}"
    )

    print()
    print(
        "Running chronological "
        "2021-2024 validation..."
    )

    validation = (
        base.run_validation(
            frame,
            feature_columns,
        )
    )

    validation_path = (
        REPORT_DIR
        / (
            "model_tournament_"
            "matchup_validation_v0_1.csv"
        )
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    summary = (
        base.summarize_validation(
            validation
        )
    )

    summary_path = (
        REPORT_DIR
        / (
            "model_tournament_"
            "matchup_summary_v0_1.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    winners = (
        base.choose_winners(
            summary
        )
    )

    print()
    print("VALIDATION WINNERS")
    print("-" * 90)

    for target in base.TARGETS:
        row = summary.loc[
            (
                summary[
                    "target"
                ].eq(target)
            )
            & (
                summary[
                    "model"
                ].eq(
                    winners[target]
                )
            )
        ].iloc[0]

        print(
            f"{target:18s}: "
            f"{winners[target]:22s} "
            f"| MAE "
            f"{row['mean_mae']:.4f}"
        )

    print()
    print(
        "RUNNING UNTOUCHED "
        "2025 HOLDOUT"
    )
    print("-" * 90)

    holdout = fit_matchup_holdout(
        frame,
        feature_columns,
        winners,
        summary,
    )

    holdout_path = (
        REPORT_DIR
        / (
            "model_tournament_"
            "matchup_holdout_2025_v0_1.csv"
        )
    )

    holdout.to_csv(
        holdout_path,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "MATCHUP-ENHANCED "
        "TOURNAMENT COMPLETE"
    )
    print("=" * 90)

    print(
        f"Validation: "
        f"{validation_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    print(
        f"Holdout: "
        f"{holdout_path}"
    )

    print()
    print(
        "Original player artifacts: "
        "UNCHANGED"
    )

    print(
        "New matchup artifacts:"
    )

    for target in base.TARGETS:
        print(
            "  "
            f"models/artifacts/"
            f"cfb_{target}_"
            "matchup_quant_v0_1.joblib"
        )


if __name__ == "__main__":
    main()
