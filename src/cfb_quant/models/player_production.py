from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from cfb_quant.models import tournament as base


ORIGINAL_DATA = Path(
    "data/processed/cfb_pregame_features_through_2026_week_3_v0_1.parquet"
)

ORIGINAL_MANIFEST = Path(
    "reports/cfb_feature_manifest_through_2026_week_3_v0_1.json"
)

MATCHUP_DATA = Path(
    "data/processed/cfb_pregame_features_matchup_through_2026_week_3_v0_1.parquet"
)

MATCHUP_MANIFEST = Path(
    "reports/cfb_feature_manifest_matchup_through_2026_week_3_v0_1.json"
)

ARTIFACT_DIR = Path(
    "models/artifacts"
)

REPORT_PATH = Path(
    "reports/cfb_player_production_training_through_2026_week_3_v0_1.csv"
)


# Frozen winners selected by the original 2021-2024 validation tournament.
ORIGINAL_WINNERS = {
    "pass_attempts": "hist_gradient_boosting",
    "completions": "hist_gradient_boosting",
    "passing_yards": "extra_trees",
    "rush_attempts": "hist_gradient_boosting",
    "rushing_yards": "hist_gradient_boosting",
    "receptions": "hist_gradient_boosting",
    "receiving_yards": "hist_gradient_boosting",
}


# Frozen winners from the matchup-enhanced validation tournament.
MATCHUP_WINNERS = {
    "pass_attempts": "hist_gradient_boosting",
    "completions": "hist_gradient_boosting",
    "passing_yards": "hist_gradient_boosting",
    "rush_attempts": "hist_gradient_boosting",
    "rushing_yards": "hist_gradient_boosting",
    "receptions": "hist_gradient_boosting",
    "receiving_yards": "hist_gradient_boosting",
}


def load_dataset(
    data_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, list[str]]:

    frame = pd.read_parquet(
        data_path
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    feature_columns = list(
        manifest["feature_columns"]
    )

    missing = [
        col
        for col in feature_columns
        if col not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{data_path}: missing "
            f"{len(missing)} manifest features. "
            f"First: {missing[:20]}"
        )

    leaks = (
        set(base.TARGETS)
        & set(feature_columns)
    )

    if leaks:
        raise RuntimeError(
            f"Direct target leakage: "
            f"{sorted(leaks)}"
        )

    return frame, feature_columns


def production_cutoff_mask(
    frame: pd.DataFrame,
) -> pd.Series:

    season = pd.to_numeric(
        frame["season"],
        errors="coerce",
    )

    week = pd.to_numeric(
        frame["week"],
        errors="coerce",
    )

    return (
        season.lt(2026)
        | (
            season.eq(2026)
            & week.le(3)
        )
    )


def train_set(
    *,
    label: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    winners: dict[str, str],
    artifact_suffix: str,
) -> list[dict]:

    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    cutoff = production_cutoff_mask(
        frame
    )

    if not cutoff.any():
        raise RuntimeError(
            f"{label}: no rows inside "
            "production cutoff."
        )

    outside = frame.loc[~cutoff]

    if not outside.empty:
        print(
            f"Rows excluded after cutoff: "
            f"{len(outside):,}"
        )

    rows = []

    for target in base.TARGETS:

        if target not in winners:
            raise RuntimeError(
                f"{label}: no frozen winner "
                f"for {target}"
            )

        model_name = winners[target]

        if model_name not in base.MODEL_FACTORIES:
            raise RuntimeError(
                f"{label}: unknown model "
                f"{model_name} for {target}"
            )

        eligible = base._target_mask(
            frame,
            target,
        )

        train_mask = (
            eligible
            & cutoff
        )

        train = frame.loc[
            train_mask
        ].copy()

        if train.empty:
            raise RuntimeError(
                f"{label}: no training rows "
                f"for {target}"
            )

        max_season = int(
            pd.to_numeric(
                train["season"],
                errors="coerce",
            ).max()
        )

        if max_season != 2026:
            raise RuntimeError(
                f"{label} {target}: expected "
                f"training through 2026, "
                f"max season is {max_season}"
            )

        train_2026 = train.loc[
            pd.to_numeric(
                train["season"],
                errors="coerce",
            ).eq(2026)
        ]

        if train_2026.empty:
            raise RuntimeError(
                f"{label} {target}: "
                "no eligible 2026 rows."
            )

        max_week_2026 = int(
            pd.to_numeric(
                train_2026["week"],
                errors="coerce",
            ).max()
        )

        if max_week_2026 != 3:
            raise RuntimeError(
                f"{label} {target}: expected "
                f"2026 cutoff Week 3, "
                f"got Week {max_week_2026}"
            )

        X_train = base._to_numeric_frame(
            train,
            feature_columns,
        )

        y_train = pd.to_numeric(
            train[target],
            errors="coerce",
        )

        model = (
            base.MODEL_FACTORIES[
                model_name
            ](
                target
            )
        )

        print()
        print(
            f"{target:18s} | "
            f"{model_name:22s} | "
            f"rows={len(train):,} | "
            f"features={len(feature_columns):,}"
        )

        model.fit(
            X_train,
            y_train,
        )

        artifact = {
            "artifact_version":
                "production_v0_1",

            "artifact_type":
                artifact_suffix,

            "target":
                target,

            "model_name":
                model_name,

            "model":
                model,

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
                    in train["season"]
                    .dropna()
                    .unique()
                ),

            "max_train_season":
                2026,

            "max_train_week_2026":
                3,

            "production_training_cutoff":
                "2026_week_3_completed",

            "eligibility_family":
                base.TARGET_FAMILY[
                    target
                ],

            "eligibility_history_columns": [
                col
                for col in base.ELIGIBILITY_HISTORY[
                    base.TARGET_FAMILY[
                        target
                    ]
                ]
                if col in frame.columns
            ],

            "prediction_bounds":
                base.PREDICTION_BOUNDS[
                    target
                ],

            "winner_selection":
                "Frozen from pre-2026 validation; "
                "not re-selected on production data.",
        }

        artifact_path = (
            ARTIFACT_DIR
            / (
                f"cfb_player_{target}_"
                f"{artifact_suffix}_v0_1.joblib"
            )
        )

        joblib.dump(
            artifact,
            artifact_path,
        )

        rows.append({
            "model_set":
                label,

            "target":
                target,

            "model_name":
                model_name,

            "training_rows":
                int(len(train)),

            "feature_count":
                len(feature_columns),

            "max_train_season":
                2026,

            "max_train_week_2026":
                3,

            "artifact":
                str(artifact_path),
        })

        print(
            f"  -> {artifact_path}"
        )

    return rows


def main() -> None:

    print("=" * 100)
    print(
        "CFB PLAYER PRODUCTION TRAINING "
        "THROUGH 2026 WEEK 3"
    )
    print("=" * 100)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_frame, original_features = (
        load_dataset(
            ORIGINAL_DATA,
            ORIGINAL_MANIFEST,
        )
    )

    matchup_frame, matchup_features = (
        load_dataset(
            MATCHUP_DATA,
            MATCHUP_MANIFEST,
        )
    )

    print(
        f"Original rows: "
        f"{len(original_frame):,}"
    )

    print(
        f"Original features: "
        f"{len(original_features):,}"
    )

    print(
        f"Matchup rows: "
        f"{len(matchup_frame):,}"
    )

    print(
        f"Matchup features: "
        f"{len(matchup_features):,}"
    )

    if len(original_features) != 465:
        raise RuntimeError(
            f"Expected 465 original features, "
            f"got {len(original_features)}"
        )

    if len(matchup_features) != 472:
        raise RuntimeError(
            f"Expected 472 matchup features, "
            f"got {len(matchup_features)}"
        )

    if len(original_frame) != len(matchup_frame):
        raise RuntimeError(
            "Original/matchup row counts differ."
        )

    rows = []

    rows.extend(
        train_set(
            label="original_465",
            frame=original_frame,
            feature_columns=original_features,
            winners=ORIGINAL_WINNERS,
            artifact_suffix="production",
        )
    )

    rows.extend(
        train_set(
            label="matchup_472",
            frame=matchup_frame,
            feature_columns=matchup_features,
            winners=MATCHUP_WINNERS,
            artifact_suffix="matchup_production",
        )
    )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "PLAYER PRODUCTION TRAINING COMPLETE"
    )
    print("=" * 100)

    print(
        f"Artifacts trained: "
        f"{len(report)}"
    )

    print()
    print(report.to_string(index=False))

    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print(
        "Evaluation artifacts: UNCHANGED"
    )


if __name__ == "__main__":
    main()
