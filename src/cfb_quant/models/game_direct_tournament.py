from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_PATH = Path(
    "data/processed/cfb_matchup_features_v0_1.parquet"
)

MANIFEST_PATH = Path(
    "reports/cfb_matchup_feature_manifest_v0_1.json"
)

BASELINE_PATH = Path(
    "reports/cfb_game_oof_calibration_v0_1.csv"
)

OOF_OUT = Path(
    "reports/cfb_direct_game_model_oof_v0_1.parquet"
)

SUMMARY_OUT = Path(
    "reports/cfb_direct_game_model_summary_v0_1.csv"
)


TEST_SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]


def make_model(name: str):

    if name == "ridge":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=100.0
                    ),
                ),
            ]
        )

    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=2.0,
                        random_state=20260923,
                    ),
                ),
            ]
        )

    if name == "extra_trees":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        max_features=0.70,
                        n_jobs=-1,
                        random_state=20260923,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


print("=" * 120)
print("CFB DIRECT GAME MODEL TOURNAMENT v0.1")
print("=" * 120)

# =====================================================================
# LOAD TEAM-GAME FEATURES
# =====================================================================

team = pd.read_parquet(
    FEATURE_PATH
)

manifest = json.loads(
    MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)

feature_columns = manifest.get(
    "feature_columns"
)

if not isinstance(
    feature_columns,
    list,
):
    raise RuntimeError(
        "Manifest does not contain feature_columns."
    )

feature_columns = [
    c
    for c in feature_columns
    if c in team.columns
]

print(
    f"Team rows:          {len(team):,}"
)

print(
    f"Manifest features:  {len(feature_columns):,}"
)

required = {
    "game_id",
    "season",
    "week",
    "season_type",
    "team",
    "opponent",
    "home_away",
    "points",
}

missing = required - set(
    team.columns
)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

# =====================================================================
# PAIR HOME + AWAY
# =====================================================================

ids = [
    "game_id",
    "season",
    "week",
    "season_type",
]

base_cols = (
    ids
    +
    [
        "team",
        "opponent",
        "points",
    ]
    +
    feature_columns
)

home = team[
    team["home_away"].eq("home")
][base_cols].copy()

away = team[
    team["home_away"].eq("away")
][base_cols].copy()

home_rename = {
    "team": "home_team",
    "opponent": "home_listed_opponent",
    "points": "actual_home_points",
}

away_rename = {
    "team": "away_team",
    "opponent": "away_listed_opponent",
    "points": "actual_away_points",
}

for col in feature_columns:
    home_rename[col] = f"home__{col}"
    away_rename[col] = f"away__{col}"

home = home.rename(
    columns=home_rename
)

away = away.rename(
    columns=away_rename
)

games = home.merge(
    away,
    on=ids,
    how="inner",
    validate="one_to_one",
)

games["pairing_ok"] = (
    (
        games["home_listed_opponent"]
        ==
        games["away_team"]
    )
    &
    (
        games["away_listed_opponent"]
        ==
        games["home_team"]
    )
)

bad_pairs = int(
    (~games["pairing_ok"]).sum()
)

if bad_pairs:
    raise RuntimeError(
        f"Bad home/away pairings: {bad_pairs}"
    )

games["actual_home_margin"] = (
    games["actual_home_points"]
    -
    games["actual_away_points"]
)

games["actual_total"] = (
    games["actual_home_points"]
    +
    games["actual_away_points"]
)

# =====================================================================
# ADD CURRENT DERIVED-POINTS BASELINE
# =====================================================================

baseline = pd.read_csv(
    BASELINE_PATH
)[
    [
        "game_id",
        "season",
        "pred_home_margin",
        "pred_total",
    ]
].drop_duplicates(
    subset=[
        "game_id",
        "season",
    ]
)

games = games.merge(
    baseline,
    on=[
        "game_id",
        "season",
    ],
    how="left",
    validate="one_to_one",
)

# =====================================================================
# GAME-LEVEL FEATURES
#
# For every leakage-safe team feature:
#
# difference = home - away
# sum        = home + away
#
# Difference features are naturally useful for margin.
# Sum features are naturally useful for total.
#
# Both targets get both sets.
#
# NO MARKET DATA IS USED.
# =====================================================================

feature_data = {}

for col in feature_columns:

    h = pd.to_numeric(
        games[f"home__{col}"],
        errors="coerce",
    )

    a = pd.to_numeric(
        games[f"away__{col}"],
        errors="coerce",
    )

    feature_data[
        f"diff__{col}"
    ] = h - a

    feature_data[
        f"sum__{col}"
    ] = h + a


X = pd.DataFrame(
    feature_data,
    index=games.index,
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

print(
    f"Paired games:        {len(games):,}"
)

print(
    f"Direct features:     {X.shape[1]:,}"
)

print(
    f"Game seasons:        "
    f"{games['season'].min()}-"
    f"{games['season'].max()}"
)

# =====================================================================
# CHRONOLOGICAL OOF
# =====================================================================

models = [
    "ridge",
    "hist_gradient_boosting",
    "extra_trees",
]

targets = {
    "margin": "actual_home_margin",
    "total": "actual_total",
}

oof_frames = []

for target_name, target_col in targets.items():

    print("\n" + "=" * 120)
    print(
        f"TARGET: {target_name.upper()}"
    )
    print("=" * 120)

    for season in TEST_SEASONS:

        train_mask = (
            (games["season"] < season)
            &
            games[target_col].notna()
        )

        test_mask = (
            games["season"].eq(season)
            &
            games[target_col].notna()
        )

        train_idx = games.index[
            train_mask
        ]

        test_idx = games.index[
            test_mask
        ]

        if len(train_idx) == 0:
            continue

        if len(test_idx) == 0:
            continue

        print(
            f"\n{season}: "
            f"train={len(train_idx):,} "
            f"test={len(test_idx):,}"
        )

        X_train = X.loc[
            train_idx
        ]

        y_train = games.loc[
            train_idx,
            target_col,
        ]

        X_test = X.loc[
            test_idx
        ]

        y_test = games.loc[
            test_idx,
            target_col,
        ]

        for model_name in models:

            print(
                f"  fitting "
                f"{model_name:<25}",
                end="",
                flush=True,
            )

            model = make_model(
                model_name
            )

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_test
            )

            frame = games.loc[
                test_idx,
                [
                    "game_id",
                    "season",
                    "week",
                    "season_type",
                    "home_team",
                    "away_team",
                ],
            ].copy()

            frame["target"] = (
                target_name
            )

            frame["model"] = (
                model_name
            )

            frame["actual"] = (
                y_test.to_numpy()
            )

            frame["prediction"] = pred

            oof_frames.append(
                frame
            )

            mae = np.mean(
                np.abs(
                    y_test.to_numpy()
                    -
                    pred
                )
            )

            rmse = np.sqrt(
                np.mean(
                    (
                        y_test.to_numpy()
                        -
                        pred
                    )
                    ** 2
                )
            )

            print(
                f" MAE={mae:.3f} "
                f"RMSE={rmse:.3f}"
            )

# =====================================================================
# ADD EXISTING DERIVED POINTS BASELINE
# =====================================================================

baseline_specs = {
    "margin": "pred_home_margin",
    "total": "pred_total",
}

for target_name, pred_col in baseline_specs.items():

    actual_col = targets[
        target_name
    ]

    mask = (
        games["season"].isin(
            TEST_SEASONS
        )
        &
        games[pred_col].notna()
        &
        games[actual_col].notna()
    )

    frame = games.loc[
        mask,
        [
            "game_id",
            "season",
            "week",
            "season_type",
            "home_team",
            "away_team",
        ],
    ].copy()

    frame["target"] = (
        target_name
    )

    frame["model"] = (
        "derived_points_ridge"
    )

    frame["actual"] = (
        games.loc[
            mask,
            actual_col,
        ].to_numpy()
    )

    frame["prediction"] = (
        games.loc[
            mask,
            pred_col,
        ].to_numpy()
    )

    oof_frames.append(
        frame
    )

# =====================================================================
# COMBINE OOF
# =====================================================================

oof = pd.concat(
    oof_frames,
    ignore_index=True,
)

oof["residual"] = (
    oof["actual"]
    -
    oof["prediction"]
)

oof["abs_error"] = (
    oof["residual"].abs()
)

oof["sq_error"] = (
    oof["residual"] ** 2
)

oof.to_parquet(
    OOF_OUT,
    index=False,
)

# =====================================================================
# SUMMARY
# =====================================================================

summary_rows = []

for target_name in targets:

    for model_name in sorted(
        oof[
            oof["target"].eq(
                target_name
            )
        ]["model"].unique()
    ):

        model_df = oof[
            oof["target"].eq(
                target_name
            )
            &
            oof["model"].eq(
                model_name
            )
        ]

        split_masks = {
            "development_2021_2024":
                model_df[
                    "season"
                ].between(
                    2021,
                    2024,
                ),

            "holdout_2025":
                model_df[
                    "season"
                ].eq(
                    2025
                ),
        }

        for split_name, mask in split_masks.items():

            sub = model_df[
                mask
            ]

            if len(sub) == 0:
                continue

            summary_rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "split": split_name,
                    "games": len(sub),

                    "mae":
                        sub[
                            "abs_error"
                        ].mean(),

                    "rmse":
                        np.sqrt(
                            sub[
                                "sq_error"
                            ].mean()
                        ),

                    "bias_actual_minus_pred":
                        sub[
                            "residual"
                        ].mean(),

                    "residual_std":
                        sub[
                            "residual"
                        ].std(
                            ddof=1
                        ),
                }
            )


summary = pd.DataFrame(
    summary_rows
)

summary[
    [
        "mae",
        "rmse",
        "bias_actual_minus_pred",
        "residual_std",
    ]
] = summary[
    [
        "mae",
        "rmse",
        "bias_actual_minus_pred",
        "residual_std",
    ]
].round(4)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)

# =====================================================================
# YEAR-BY-YEAR
# =====================================================================

print("\n" + "=" * 120)
print("YEAR-BY-YEAR RESULTS")
print("=" * 120)

year_rows = []

for (
    target_name,
    model_name,
    season
), sub in oof.groupby(
    [
        "target",
        "model",
        "season",
    ]
):

    year_rows.append(
        {
            "target": target_name,
            "model": model_name,
            "season": season,
            "games": len(sub),
            "mae":
                sub[
                    "abs_error"
                ].mean(),
            "rmse":
                np.sqrt(
                    sub[
                        "sq_error"
                    ].mean()
                ),
        }
    )

year = pd.DataFrame(
    year_rows
)

print(
    year.round(3).to_string(
        index=False
    )
)

# =====================================================================
# DEVELOPMENT RANKING
# =====================================================================

print("\n" + "=" * 120)
print("DEVELOPMENT 2021-2024 RANKING")
print("=" * 120)

dev = summary[
    summary["split"].eq(
        "development_2021_2024"
    )
].sort_values(
    [
        "target",
        "rmse",
        "mae",
    ]
)

print(
    dev[
        [
            "target",
            "model",
            "games",
            "mae",
            "rmse",
            "bias_actual_minus_pred",
        ]
    ].to_string(
        index=False
    )
)

# =====================================================================
# 2025 HOLDOUT
# =====================================================================

print("\n" + "=" * 120)
print("UNTOUCHED 2025 HOLDOUT")
print("=" * 120)

hold = summary[
    summary["split"].eq(
        "holdout_2025"
    )
].sort_values(
    [
        "target",
        "rmse",
        "mae",
    ]
)

print(
    hold[
        [
            "target",
            "model",
            "games",
            "mae",
            "rmse",
            "bias_actual_minus_pred",
        ]
    ].to_string(
        index=False
    )
)

# =====================================================================
# DEV WINNERS -> HOLDOUT COMPARISON
# =====================================================================

print("\n" + "=" * 120)
print("DEVELOPMENT WINNER -> 2025 TEST")
print("=" * 120)

for target_name in [
    "margin",
    "total",
]:

    target_dev = dev[
        dev["target"].eq(
            target_name
        )
    ]

    winner = (
        target_dev.iloc[0][
            "model"
        ]
    )

    print(
        f"\n{target_name.upper()} "
        f"development winner: "
        f"{winner}"
    )

    comparison = summary[
        summary["target"].eq(
            target_name
        )
        &
        summary["model"].isin(
            [
                winner,
                "derived_points_ridge",
            ]
        )
    ][
        [
            "model",
            "split",
            "games",
            "mae",
            "rmse",
            "bias_actual_minus_pred",
        ]
    ]

    print(
        comparison.to_string(
            index=False
        )
    )


print("\n" + "=" * 120)
print("OUTPUTS")
print("=" * 120)

print(
    "OOF predictions:",
    OOF_OUT,
)

print(
    "Summary:",
    SUMMARY_OUT,
)

print("=" * 120)
print(
    "MARKET-BLIND GUARD: "
    "no sportsbook lines were used as model inputs."
)
print("=" * 120)

