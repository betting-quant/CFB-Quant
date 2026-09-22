from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cfb_quant.models.matchup_tournament import (
    MIN_TRAIN_SEASON,
    HOLDOUT_YEAR,
    MODEL_FACTORIES,
    TARGETS,
    _fit_and_predict,
    _load_inputs,
    _target_mask,
    _to_numeric_frame,
)


DEFAULT_DATA = "data/processed/cfb_matchup_features_v0_1.parquet"
DEFAULT_MANIFEST = "reports/cfb_matchup_feature_manifest_v0_1.json"

DEFAULT_LONG_OUTPUT = (
    "data/processed/"
    "cfb_matchup_oof_predictions_v0_1.parquet"
)

DEFAULT_WIDE_OUTPUT = (
    "data/processed/"
    "cfb_matchup_oof_features_v0_1.parquet"
)


def build_oof_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    prediction_years = list(
        range(
            MIN_TRAIN_SEASON + 1,
            HOLDOUT_YEAR + 1,
        )
    )

    for target in TARGETS:
        eligible = _target_mask(
            frame,
            target,
        )

        for prediction_year in prediction_years:
            train_mask = (
                eligible
                & frame["season"].between(
                    MIN_TRAIN_SEASON,
                    prediction_year - 1,
                )
            )

            predict_mask = (
                eligible
                & frame["season"].eq(
                    prediction_year
                )
            )

            train = frame.loc[
                train_mask
            ].copy()

            predict = frame.loc[
                predict_mask
            ].copy()

            if train.empty or predict.empty:
                print(
                    f"SKIP {target:20s} | "
                    f"{prediction_year} | "
                    f"train={len(train):,} "
                    f"predict={len(predict):,}"
                )
                continue

            if int(train["season"].max()) >= prediction_year:
                raise RuntimeError(
                    "Chronological leakage detected: "
                    f"{target} {prediction_year}"
                )

            X_train = _to_numeric_frame(
                train,
                feature_columns,
            )

            y_train = pd.to_numeric(
                train[target],
                errors="coerce",
            )

            X_predict = _to_numeric_frame(
                predict,
                feature_columns,
            )

            # Early OOF folds can contain features that are entirely missing
            # or have only one observed training value. Such columns contain
            # no predictive information in that fold and can break histogram
            # binning in some sklearn versions.
            usable_feature_columns = [
                col
                for col in feature_columns
                if X_train[col].notna().any()
                and X_train[col].nunique(dropna=True) >= 2
            ]

            if not usable_feature_columns:
                raise RuntimeError(
                    f"No usable matchup features for "
                    f"{target} {prediction_year}."
                )

            X_train = X_train[
                usable_feature_columns
            ]

            X_predict = X_predict[
                usable_feature_columns
            ]

            print(
                f"  usable features: "
                f"{len(usable_feature_columns)}/"
                f"{len(feature_columns)}"
            )

            for model_name in MODEL_FACTORIES:
                _, pred = _fit_and_predict(
                    model_name,
                    target,
                    X_train,
                    y_train,
                    X_predict,
                )

                out = predict[
                    [
                        "game_id",
                        "season",
                        "week",
                        "season_type",
                        "team",
                        "opponent",
                        "home_away",
                    ]
                ].copy()

                out["target"] = target
                out["model"] = model_name
                out["prediction"] = pred
                out["actual"] = pd.to_numeric(
                    predict[target],
                    errors="coerce",
                ).to_numpy()

                out["train_start_season"] = int(
                    train["season"].min()
                )

                out["train_end_season"] = int(
                    train["season"].max()
                )

                rows.append(
                    out
                )

                print(
                    f"{target:20s} | "
                    f"{prediction_year} | "
                    f"{model_name:22s} | "
                    f"train through "
                    f"{int(train['season'].max())} | "
                    f"rows {len(predict):,}"
                )

    if not rows:
        raise RuntimeError(
            "No matchup OOF predictions generated."
        )

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    duplicate_keys = [
        "game_id",
        "team",
        "target",
        "model",
    ]

    dupes = result.duplicated(
        duplicate_keys,
        keep=False,
    )

    if dupes.any():
        raise RuntimeError(
            "Duplicate matchup OOF predictions "
            "detected."
        )

    bad_chronology = (
        result["train_end_season"]
        >= result["season"]
    )

    if bad_chronology.any():
        raise RuntimeError(
            "OOF prediction chronology guard failed."
        )

    return result


def build_wide_features(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    wide_source = predictions[
        [
            "game_id",
            "season",
            "week",
            "season_type",
            "team",
            "opponent",
            "home_away",
            "target",
            "model",
            "prediction",
        ]
    ].copy()

    wide_source["feature_name"] = (
        "matchup_"
        + wide_source["model"].astype(str)
        + "_"
        + wide_source["target"].astype(str)
    )

    index_cols = [
        "game_id",
        "season",
        "week",
        "season_type",
        "team",
        "opponent",
        "home_away",
    ]

    wide = (
        wide_source.pivot(
            index=index_cols,
            columns="feature_name",
            values="prediction",
        )
        .reset_index()
    )

    wide.columns.name = None

    feature_cols = sorted(
        c
        for c in wide.columns
        if c.startswith("matchup_")
    )

    wide = wide[
        index_cols
        + feature_cols
    ]

    if wide.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate team-game rows "
            "in wide matchup feature table."
        )

    return wide


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe chronological "
            "CFB matchup OOF predictions."
        )
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--long-output",
        default=DEFAULT_LONG_OUTPUT,
    )

    parser.add_argument(
        "--wide-output",
        default=DEFAULT_WIDE_OUTPUT,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CFB MATCHUP OOF PREDICTIONS v0.1")
    print("=" * 80)

    frame, feature_columns = _load_inputs(
        args.data,
        args.manifest,
    )

    predictions = build_oof_predictions(
        frame,
        feature_columns,
    )

    long_path = Path(
        args.long_output
    )

    long_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        long_path,
        index=False,
    )

    wide = build_wide_features(
        predictions
    )

    wide_path = Path(
        args.wide_output
    )

    wide.to_parquet(
        wide_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("OOF MATCHUP PREDICTIONS COMPLETE")
    print("=" * 80)

    print(
        "Long rows:",
        f"{len(predictions):,}",
    )

    print(
        "Wide team-game rows:",
        f"{len(wide):,}",
    )

    matchup_features = [
        c
        for c in wide.columns
        if c.startswith("matchup_")
    ]

    print(
        "Matchup projection features:",
        len(matchup_features),
    )

    print(
        "Prediction seasons:",
        f"{int(predictions['season'].min())}-"
        f"{int(predictions['season'].max())}",
    )

    print(
        "Long output:",
        long_path,
    )

    print(
        "Wide output:",
        wide_path,
    )

    print()
    print(
        "CHRONOLOGY GUARD: every prediction "
        "was trained only on prior seasons."
    )


if __name__ == "__main__":
    main()
