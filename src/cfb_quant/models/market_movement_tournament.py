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

QUANT_PATH = Path(
    "reports/cfb_game_oof_calibration_v0_1.csv"
)

DIRECT_PATH = Path(
    "reports/cfb_direct_game_model_oof_v0_1.parquet"
)

MARKET_PATH = Path(
    "reports/cfb_historical_market_open_close_v0_2.csv"
)

OOF_OUT = Path(
    "reports/cfb_market_movement_oof_v0_1.parquet"
)

SUMMARY_OUT = Path(
    "reports/cfb_market_movement_summary_v0_1.csv"
)


# =====================================================================
# MODELS
# =====================================================================

def make_model(name):

    if name == "ridge":

        return Pipeline([
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
        ])

    if name == "hist_gradient_boosting":

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=5.0,
                    random_state=20260923,
                ),
            ),
        ])

    if name == "extra_trees":

        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=5,
                    max_features=0.60,
                    n_jobs=-1,
                    random_state=20260923,
                ),
            ),
        ])

    raise ValueError(name)


print("=" * 120)
print("CFB MARKET MOVEMENT TOURNAMENT v0.1")
print("=" * 120)


# =====================================================================
# FOOTBALL FEATURES
# =====================================================================

team = pd.read_parquet(
    FEATURE_PATH
)

manifest = json.loads(
    MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
)

feature_columns = [
    c
    for c in manifest["feature_columns"]
    if c in team.columns
]

ids = [
    "game_id",
    "season",
    "week",
    "season_type",
]

cols = (
    ids
    + [
        "team",
        "opponent",
    ]
    + feature_columns
)


home = team[
    team["home_away"].eq("home")
][cols].copy()

away = team[
    team["home_away"].eq("away")
][cols].copy()


home_rename = {
    "team": "home_team",
    "opponent": "home_opponent",
}

away_rename = {
    "team": "away_team",
    "opponent": "away_opponent",
}


for col in feature_columns:

    home_rename[col] = (
        f"home__{col}"
    )

    away_rename[col] = (
        f"away__{col}"
    )


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


# =====================================================================
# MARKET-BLIND QUANT
# =====================================================================

quant = pd.read_csv(
    QUANT_PATH
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
    quant,
    on=[
        "game_id",
        "season",
    ],
    how="inner",
    validate="one_to_one",
)


# =====================================================================
# DIRECT EXTRA TREES FAIR MARGIN
# =====================================================================

direct = pd.read_parquet(
    DIRECT_PATH
)

direct = direct[
    direct["target"].eq("margin")
    &
    direct["model"].eq("extra_trees")
][
    [
        "game_id",
        "season",
        "prediction",
    ]
].copy()

direct = direct.rename(
    columns={
        "prediction":
            "direct_margin"
    }
)


games = games.merge(
    direct,
    on=[
        "game_id",
        "season",
    ],
    how="inner",
    validate="one_to_one",
)


# =====================================================================
# OPEN / CLOSE MARKET
# =====================================================================

market = pd.read_csv(
    MARKET_PATH
)

market = market.drop(
    columns=["week"],
    errors="ignore",
)

market = market[
    [
        "game_id",
        "season",

        "open_home_spread",
        "close_home_spread",

        "open_spread_providers",
        "close_spread_providers",

        "open_spread_book_range",
        "close_spread_book_range",

        "open_total",
        "close_total",

        "open_total_providers",
        "close_total_providers",
    ]
]


games = games.merge(
    market,
    on=[
        "game_id",
        "season",
    ],
    how="inner",
    validate="one_to_one",
)


# =====================================================================
# SIGN CONVENTION
# =====================================================================

games["open_home_margin"] = (
    -games["open_home_spread"]
)

games["close_home_margin"] = (
    -games["close_home_spread"]
)


# =====================================================================
# MOVEMENT TARGETS
#
# Positive spread movement:
# market moved toward HOME
#
# Positive total movement:
# total moved UP
# =====================================================================

games["spread_move"] = (
    games["close_home_margin"]
    -
    games["open_home_margin"]
)

games["total_move"] = (
    games["close_total"]
    -
    games["open_total"]
)


# =====================================================================
# QUANT DISAGREEMENT
# =====================================================================

games["old_spread_edge"] = (
    games["pred_home_margin"]
    -
    games["open_home_margin"]
)

games["direct_spread_edge"] = (
    games["direct_margin"]
    -
    games["open_home_margin"]
)

games["total_edge"] = (
    games["pred_total"]
    -
    games["open_total"]
)


games["old_spread_abs_edge"] = (
    games["old_spread_edge"].abs()
)

games["direct_spread_abs_edge"] = (
    games["direct_spread_edge"].abs()
)

games["total_abs_edge"] = (
    games["total_edge"].abs()
)


# =====================================================================
# FOOTBALL GAME FEATURES
# =====================================================================

feature_data = {}

for col in feature_columns:

    h = pd.to_numeric(
        games[
            f"home__{col}"
        ],
        errors="coerce",
    )

    a = pd.to_numeric(
        games[
            f"away__{col}"
        ],
        errors="coerce",
    )

    feature_data[
        f"diff__{col}"
    ] = h - a

    feature_data[
        f"sum__{col}"
    ] = h + a


X_football = pd.DataFrame(
    feature_data,
    index=games.index,
)


# =====================================================================
# OPENING-TIME FEATURES ONLY
# =====================================================================

X_market = pd.DataFrame(
    index=games.index
)

X_market["week"] = games["week"]

X_market["open_home_margin"] = (
    games["open_home_margin"]
)

X_market["open_total"] = (
    games["open_total"]
)

X_market["open_spread_providers"] = (
    games["open_spread_providers"]
)

X_market["open_total_providers"] = (
    games["open_total_providers"]
)

X_market["open_spread_book_range"] = (
    games["open_spread_book_range"]
)

X_market["pred_home_margin"] = (
    games["pred_home_margin"]
)

X_market["direct_margin"] = (
    games["direct_margin"]
)

X_market["pred_total"] = (
    games["pred_total"]
)

X_market["old_spread_edge"] = (
    games["old_spread_edge"]
)

X_market["direct_spread_edge"] = (
    games["direct_spread_edge"]
)

X_market["total_edge"] = (
    games["total_edge"]
)

X_market["old_spread_abs_edge"] = (
    games["old_spread_abs_edge"]
)

X_market["direct_spread_abs_edge"] = (
    games["direct_spread_abs_edge"]
)

X_market["total_abs_edge"] = (
    games["total_abs_edge"]
)


X = pd.concat(
    [
        X_football,
        X_market,
    ],
    axis=1,
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)


print(
    f"Paired games:    {len(games):,}"
)

print(
    f"Model features:  {X.shape[1]:,}"
)


# =====================================================================
# OOF
# =====================================================================

models = [
    "ridge",
    "hist_gradient_boosting",
    "extra_trees",
]

targets = {
    "spread":
        "spread_move",

    "total":
        "total_move",
}


oof_frames = []


for market_name, target_col in targets.items():

    print("\n" + "=" * 120)

    print(
        f"TARGET: {market_name.upper()} OPEN -> CLOSE MOVEMENT"
    )

    print("=" * 120)


    for season in [
        2022,
        2023,
        2024,
        2025,
    ]:

        train_mask = (
            games["season"].lt(
                season
            )
            &
            games[target_col].notna()
        )

        test_mask = (
            games["season"].eq(
                season
            )
            &
            games[target_col].notna()
        )


        train_idx = games.index[
            train_mask
        ]

        test_idx = games.index[
            test_mask
        ]


        if (
            len(train_idx) == 0
            or len(test_idx) == 0
        ):
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

            model = make_model(
                model_name
            )

            print(
                f"  fitting "
                f"{model_name:<25}",
                end="",
                flush=True,
            )


            model.fit(
                X_train,
                y_train,
            )


            pred = model.predict(
                X_test
            )


            baseline_mae = np.mean(
                np.abs(
                    y_test.to_numpy()
                )
            )

            model_mae = np.mean(
                np.abs(
                    y_test.to_numpy()
                    -
                    pred
                )
            )


            baseline_rmse = np.sqrt(
                np.mean(
                    y_test.to_numpy()
                    ** 2
                )
            )

            model_rmse = np.sqrt(
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
                f" zero-MAE={baseline_mae:.3f}"
                f" model-MAE={model_mae:.3f}"
                f" zero-RMSE={baseline_rmse:.3f}"
                f" model-RMSE={model_rmse:.3f}"
            )


            frame = games.loc[
                test_idx,
                [
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "open_home_margin",
                    "close_home_margin",
                    "open_total",
                    "close_total",
                ],
            ].copy()


            frame["market"] = (
                market_name
            )

            frame["model"] = (
                model_name
            )

            frame["actual_move"] = (
                y_test.to_numpy()
            )

            frame["pred_move"] = pred


            oof_frames.append(
                frame
            )


oof = pd.concat(
    oof_frames,
    ignore_index=True,
)

oof.to_parquet(
    OOF_OUT,
    index=False,
)


# =====================================================================
# SUMMARY HELPER
# =====================================================================

def evaluate(sub):

    actual = sub[
        "actual_move"
    ].to_numpy()

    pred = sub[
        "pred_move"
    ].to_numpy()


    baseline_mae = np.mean(
        np.abs(actual)
    )

    model_mae = np.mean(
        np.abs(
            actual - pred
        )
    )


    baseline_rmse = np.sqrt(
        np.mean(
            actual ** 2
        )
    )

    model_rmse = np.sqrt(
        np.mean(
            (
                actual - pred
            )
            ** 2
        )
    )


    corr = (
        np.corrcoef(
            actual,
            pred,
        )[0, 1]
        if len(actual) > 1
        else np.nan
    )


    direction_rows = (
        np.abs(actual)
        >= 0.5
    )

    if direction_rows.sum():

        direction_accuracy = np.mean(
            np.sign(
                actual[
                    direction_rows
                ]
            )
            ==
            np.sign(
                pred[
                    direction_rows
                ]
            )
        )

    else:

        direction_accuracy = np.nan


    move1 = (
        np.abs(actual)
        >= 1.0
    )

    if move1.sum():

        direction_accuracy_1plus = (
            np.mean(
                np.sign(
                    actual[
                        move1
                    ]
                )
                ==
                np.sign(
                    pred[
                        move1
                    ]
                )
            )
        )

    else:

        direction_accuracy_1plus = (
            np.nan
        )


    return {
        "games":
            len(sub),

        "zero_move_mae":
            baseline_mae,

        "model_mae":
            model_mae,

        "mae_gain":
            baseline_mae
            -
            model_mae,

        "zero_move_rmse":
            baseline_rmse,

        "model_rmse":
            model_rmse,

        "rmse_gain":
            baseline_rmse
            -
            model_rmse,

        "move_corr":
            corr,

        "direction_accuracy_move_0_5plus":
            direction_accuracy,

        "direction_accuracy_move_1plus":
            direction_accuracy_1plus,

        "actual_move_mean":
            np.mean(actual),

        "pred_move_mean":
            np.mean(pred),
    }


# =====================================================================
# DEVELOPMENT 2022-2024 — ALL WEEKS
# =====================================================================

print("\n" + "=" * 120)
print("DEVELOPMENT 2022-2024 — ALL WEEKS")
print("=" * 120)


dev_rows = []


for (
    market_name,
    model_name,
), sub in oof[
    oof["season"].between(
        2022,
        2024,
    )
].groupby(
    [
        "market",
        "model",
    ]
):

    dev_rows.append(
        {
            "market":
                market_name,

            "model":
                model_name,

            **evaluate(sub),
        }
    )


dev = pd.DataFrame(
    dev_rows
)


print(
    dev.sort_values(
        [
            "market",
            "model_rmse",
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


# =====================================================================
# DEVELOPMENT 2022-2024 — WEEKS 1-4
#
# This is our current live use case.
# Winner selection is based on THIS table.
# =====================================================================

print("\n" + "=" * 120)
print("DEVELOPMENT 2022-2024 — WEEKS 1-4")
print("=" * 120)


early_dev_rows = []


early_dev_oof = oof[
    oof["season"].between(
        2022,
        2024,
    )
    &
    oof["week"].le(4)
]


for (
    market_name,
    model_name,
), sub in early_dev_oof.groupby(
    [
        "market",
        "model",
    ]
):

    early_dev_rows.append(
        {
            "market":
                market_name,

            "model":
                model_name,

            **evaluate(sub),
        }
    )


early_dev = pd.DataFrame(
    early_dev_rows
)


print(
    early_dev.sort_values(
        [
            "market",
            "model_rmse",
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


# =====================================================================
# SELECT EARLY-SEASON DEVELOPMENT WINNERS
# =====================================================================

winners = {}


for market_name in [
    "spread",
    "total",
]:

    x = early_dev[
        early_dev[
            "market"
        ].eq(
            market_name
        )
    ].sort_values(
        [
            "model_rmse",
            "model_mae",
        ]
    )


    winners[
        market_name
    ] = x.iloc[0][
        "model"
    ]


print(
    "\nEarly-season development winners:"
)

print(
    winners
)


# =====================================================================
# 2025 WEEKS 1-4 FIXED TEST
# =====================================================================

print("\n" + "=" * 120)
print("2025 WEEKS 1-4 FIXED TEST")
print("=" * 120)


hold_rows = []


for market_name, winner in winners.items():

    sub = oof[
        oof["season"].eq(
            2025
        )
        &
        oof["week"].le(4)
        &
        oof["market"].eq(
            market_name
        )
        &
        oof["model"].eq(
            winner
        )
    ]


    hold_rows.append(
        {
            "market":
                market_name,

            "winner":
                winner,

            **evaluate(sub),
        }
    )


hold = pd.DataFrame(
    hold_rows
)


print(
    hold.round(4)
    .to_string(
        index=False
    )
)


# =====================================================================
# WINNER YEAR BY YEAR — EARLY SEASON
# =====================================================================

print("\n" + "=" * 120)
print("EARLY-SEASON WINNER YEAR-BY-YEAR")
print("=" * 120)


year_rows = []


for market_name, winner in winners.items():

    for season in [
        2022,
        2023,
        2024,
        2025,
    ]:

        sub = oof[
            oof["season"].eq(
                season
            )
            &
            oof["week"].le(4)
            &
            oof["market"].eq(
                market_name
            )
            &
            oof["model"].eq(
                winner
            )
        ]


        if len(sub) == 0:
            continue


        year_rows.append(
            {
                "market":
                    market_name,

                "model":
                    winner,

                "season":
                    season,

                **evaluate(sub),
            }
        )


year = pd.DataFrame(
    year_rows
)


print(
    year.round(4)
    .to_string(
        index=False
    )
)


# =====================================================================
# SAVE SUMMARY
# =====================================================================

summary = pd.concat(
    [
        dev.assign(
            segment=
                "development_all_weeks"
        ),

        early_dev.assign(
            segment=
                "development_weeks_1_4"
        ),

        hold.assign(
            segment=
                "holdout_2025_weeks_1_4"
        ),
    ],
    ignore_index=True,
    sort=False,
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


print("\n" + "=" * 120)
print("OUTPUTS")
print("=" * 120)

print(
    "OOF:",
    OOF_OUT,
)

print(
    "Summary:",
    SUMMARY_OUT,
)

print("=" * 120)

print(
    "LEAKAGE GUARD:"
)

print(
    "Closing lines are prediction targets only."
)

print(
    "No closing line value is included in X."
)

print("=" * 120)
