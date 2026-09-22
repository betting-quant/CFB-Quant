from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from cfb_quant.models.matchup_tournament import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    MODEL_FACTORIES,
    PREDICTION_BOUNDS,
    TARGETS,
    _target_mask,
    _to_numeric_frame,
)


ARTIFACT_DIR = Path("models/artifacts")
REPORT_DIR = Path("reports")

PRODUCTION_WINNERS = {
    "points": "ridge",
    "pass_attempts": "extra_trees",
    "rush_attempts": "hist_gradient_boosting",
    "total_plays": "hist_gradient_boosting",
    "passing_yards": "hist_gradient_boosting",
    "rushing_yards": "ridge",
    "possession_seconds": "hist_gradient_boosting",
}


def main() -> None:
    print("=" * 90)
    print("CFB MATCHUP PRODUCTION TRAINER v0.1")
    print("=" * 90)

    frame = pd.read_parquet(DEFAULT_DATA)

    manifest = json.loads(
        Path(DEFAULT_MANIFEST).read_text(
            encoding="utf-8"
        )
    )

    feature_columns = list(
        manifest["feature_columns"]
    )

    missing = sorted(
        set(feature_columns) - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing matchup features: "
            + ", ".join(missing[:20])
        )

    season_numeric = pd.to_numeric(
        frame["season"],
        errors="coerce",
    )

    print(
        f"Rows available: {len(frame):,}"
    )
    print(
        f"Games available: "
        f"{frame['game_id'].nunique():,}"
    )
    print(
        f"Feature columns: "
        f"{len(feature_columns):,}"
    )

    current_2026 = frame.loc[
        season_numeric.eq(2026)
    ].copy()

    if current_2026.empty:
        raise RuntimeError(
            "No 2026 completed matchup rows found."
        )

    max_week_2026 = int(
        pd.to_numeric(
            current_2026["week"],
            errors="coerce",
        ).max()
    )

    print(
        f"2026 completed team-game rows: "
        f"{len(current_2026):,}"
    )
    print(
        f"2026 completed games: "
        f"{current_2026['game_id'].nunique():,}"
    )
    print(
        f"2026 maximum completed week: "
        f"{max_week_2026}"
    )

    if max_week_2026 != 3:
        raise RuntimeError(
            "Expected live production cutoff at "
            f"2026 Week 3, found Week {max_week_2026}."
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_rows = []

    for target in TARGETS:
        print()
        print("-" * 90)
        print(target.upper())
        print("-" * 90)

        if target not in PRODUCTION_WINNERS:
            raise RuntimeError(
                f"No production winner for {target}"
            )

        model_name = PRODUCTION_WINNERS[target]

        eligible = _target_mask(
            frame,
            target,
        )

        train = frame.loc[
            eligible
            & season_numeric.le(2026)
        ].copy()

        if train.empty:
            raise RuntimeError(
                f"No production training rows for {target}"
            )

        train_seasons = sorted(
            pd.to_numeric(
                train["season"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        X_train = _to_numeric_frame(
            train,
            feature_columns,
        )

        y_train = pd.to_numeric(
            train[target],
            errors="coerce",
        )

        if y_train.isna().any():
            raise RuntimeError(
                f"NaN target values survived "
                f"eligibility mask for {target}"
            )

        model = MODEL_FACTORIES[
            model_name
        ](
            target
        )

        print(
            f"Model: {model_name}"
        )
        print(
            f"Training rows: {len(train):,}"
        )
        print(
            f"Training seasons: "
            f"{train_seasons[0]}-{train_seasons[-1]}"
        )

        model.fit(
            X_train,
            y_train,
        )

        evaluation_path = (
            ARTIFACT_DIR
            / (
                f"cfb_matchup_{target}_"
                "quant_v0_1.joblib"
            )
        )

        if not evaluation_path.exists():
            raise RuntimeError(
                "Missing evaluation artifact: "
                f"{evaluation_path}"
            )

        evaluation_artifact = joblib.load(
            evaluation_path
        )

        if (
            evaluation_artifact.get("model_name")
            != model_name
        ):
            raise RuntimeError(
                f"Winner mismatch for {target}: "
                f"production={model_name}, "
                f"evaluation="
                f"{evaluation_artifact.get('model_name')}"
            )

        artifact = {
            "artifact_version": "v0.1",
            "artifact_type":
                "cfb_matchup_production_model",
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
                train_seasons,
            "max_train_season":
                2026,
            "max_train_week_2026":
                max_week_2026,
            "production_cutoff":
                "2026_week_3_completed",
            "prediction_bounds":
                PREDICTION_BOUNDS[target],

            # Preserve evaluation evidence from the
            # locked tournament artifact.
            "selection_validation_mean_mae":
                evaluation_artifact.get(
                    "validation_mean_mae"
                ),
            "selection_validation_mean_rmse":
                evaluation_artifact.get(
                    "validation_mean_rmse"
                ),
            "evaluation_holdout_year":
                evaluation_artifact.get(
                    "holdout_year"
                ),
            "evaluation_holdout_mae":
                evaluation_artifact.get(
                    "holdout_mae"
                ),
            "evaluation_holdout_rmse":
                evaluation_artifact.get(
                    "holdout_rmse"
                ),
            "evaluation_holdout_bias":
                evaluation_artifact.get(
                    "holdout_bias"
                ),
            "evaluation_holdout_corr":
                evaluation_artifact.get(
                    "holdout_corr"
                ),
            "model_family_selected_before_2025":
                True,
            "2025_used_for_model_selection":
                False,
            "2026_used_for_model_selection":
                False,
        }

        output_path = (
            ARTIFACT_DIR
            / (
                f"cfb_matchup_{target}_"
                "production_v0_1.joblib"
            )
        )

        joblib.dump(
            artifact,
            output_path,
        )

        print(
            f"Saved: {output_path}"
        )

        report_rows.append(
            {
                "target": target,
                "model_name": model_name,
                "training_rows":
                    int(len(train)),
                "train_start_season":
                    int(train_seasons[0]),
                "train_end_season":
                    int(train_seasons[-1]),
                "max_train_week_2026":
                    max_week_2026,
                "feature_count":
                    len(feature_columns),
                "artifact":
                    str(output_path),
            }
        )

    report = pd.DataFrame(
        report_rows
    )

    report_path = (
        REPORT_DIR
        / "cfb_matchup_production_v0_1.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print()
    print("=" * 90)
    print("MATCHUP PRODUCTION TRAINING COMPLETE")
    print("=" * 90)
    print(
        report[
            [
                "target",
                "model_name",
                "training_rows",
                "max_train_week_2026",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        f"Report: {report_path}"
    )
    print()
    print(
        "EVALUATION ARTIFACTS: unchanged"
    )
    print(
        "PRODUCTION CUTOFF: completed 2026 Week 3"
    )


if __name__ == "__main__":
    main()
