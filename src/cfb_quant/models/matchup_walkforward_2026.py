from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cfb_quant.models.matchup_tournament import (
    TARGETS,
    _make_ridge,
    _make_hgb,
    _make_extra_trees,
)


TEAM_HISTORY_PATH = Path(
    "data/processed/team_games_2026.parquet"
)

FULL_MATCHUP_HISTORY_PATH = Path(
    "data/processed/cfb_matchup_features_v0_1.parquet"
)

OUTPUT = Path(
    "data/processed/"
    "cfb_matchup_walkforward_2026_weeks_1_3_v0_1.parquet"
)

REPORT = Path(
    "reports/"
    "cfb_matchup_walkforward_2026_weeks_1_3_v0_1.csv"
)

WINNERS = {
    "points": "ridge",
    "pass_attempts": "extra_trees",
    "rush_attempts": "hist_gradient_boosting",
    "total_plays": "hist_gradient_boosting",
    "passing_yards": "hist_gradient_boosting",
    "rushing_yards": "ridge",
    "possession_seconds": "hist_gradient_boosting",
}


def main():

    full = pd.read_parquet(
        FULL_MATCHUP_HISTORY_PATH
    )

    team_2026 = pd.read_parquet(
        TEAM_HISTORY_PATH
    )

    feature_cols = [
        c for c in full.columns
        if c.startswith("off_")
        or c.startswith("def_")
        or c in {
            "is_home",
            "is_postseason",
            "week_num",
        }
    ]

    outputs = []

    print("=" * 110)
    print("2026 WEEKLY WALK-FORWARD MATCHUP PROJECTIONS")
    print("=" * 110)

    for week in [1, 2, 3]:

        print()
        print(f"WEEK {week}")
        print("-" * 110)

        train = full.loc[
            (
                full["season"] < 2026
            )
            |
            (
                full["season"].eq(2026)
                & full["week"].lt(week)
            )
        ].copy()

        test = full.loc[
            full["season"].eq(2026)
            & full["week"].eq(week)
        ].copy()

        if test.empty:
            print("No rows found.")
            continue

        result = test[
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
                "home_away",
            ]
        ].copy()

        X_train = (
            train[feature_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        X_test = (
            test[feature_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        for target in TARGETS:

            eligible = pd.to_numeric(
                train[target],
                errors="coerce",
            ).notna()

            X_fit = X_train.loc[
                eligible
            ]

            y_fit = pd.to_numeric(
                train.loc[
                    eligible,
                    target,
                ],
                errors="coerce",
            )

            model_name = WINNERS[target]

            model_factories = {
                "ridge": _make_ridge,
                "hist_gradient_boosting": _make_hgb,
                "extra_trees": _make_extra_trees,
            }

            model = model_factories[
                model_name
            ](
                target
            )

            model.fit(
                X_fit,
                y_fit,
            )

            pred = model.predict(
                X_test
            )

            result[
                f"matchup_proj_{target}"
            ] = np.asarray(
                pred,
                dtype=float,
            )

            print(
                f"{target}: "
                f"{model_name} "
                f"train={len(X_fit):,} "
                f"test={len(X_test):,}"
            )

        outputs.append(
            result
        )

    out = pd.concat(
        outputs,
        ignore_index=True,
    )

    if out.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate team-game rows found."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_parquet(
        OUTPUT,
        index=False,
    )

    out.to_csv(
        REPORT,
        index=False,
    )

    print()
    print("=" * 110)
    print("WALK-FORWARD COMPLETE")
    print("=" * 110)

    print(
        out.groupby(
            "week"
        ).size()
    )

    print()
    print("Rows:", len(out))
    print(
        "Games:",
        out["game_id"].nunique(),
    )

    print()
    print("Saved:", OUTPUT)
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()


