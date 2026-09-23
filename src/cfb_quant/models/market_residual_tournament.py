from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_PATH = Path("data/processed/cfb_matchup_features_v0_1.parquet")
MANIFEST_PATH = Path("reports/cfb_matchup_feature_manifest_v0_1.json")
OOF_MODEL_PATH = Path("reports/cfb_game_oof_calibration_v0_1.csv")
MARKET_PATH = Path("reports/cfb_historical_market_open_close_v0_2.csv")

OOF_OUT = Path("reports/cfb_market_residual_oof_v0_1.parquet")
SUMMARY_OUT = Path("reports/cfb_market_residual_summary_v0_1.csv")


def make_model(name):

    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=100.0)),
        ])

    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
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
            ("imputer", SimpleImputer(strategy="median")),
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
print("CFB MARKET RESIDUAL MODEL TOURNAMENT v0.1")
print("=" * 120)


# =====================================================================
# LOAD LEAKAGE-SAFE FOOTBALL FEATURES
# =====================================================================

team = pd.read_parquet(FEATURE_PATH)

manifest = json.loads(
    MANIFEST_PATH.read_text(encoding="utf-8")
)

feature_columns = [
    c for c in manifest["feature_columns"]
    if c in team.columns
]

print(f"Team rows:         {len(team):,}")
print(f"Football features: {len(feature_columns):,}")


# =====================================================================
# HOME/AWAY GAME PAIRING
# =====================================================================

ids = [
    "game_id",
    "season",
    "week",
    "season_type",
]

cols = (
    ids
    + ["team", "opponent", "points"]
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

home = home.rename(columns=home_rename)
away = away.rename(columns=away_rename)

games = home.merge(
    away,
    on=ids,
    how="inner",
    validate="one_to_one",
)

games["actual_home_margin"] = (
    games["actual_home_points"]
    - games["actual_away_points"]
)

games["actual_total"] = (
    games["actual_home_points"]
    + games["actual_away_points"]
)


# =====================================================================
# ADD MARKET-BLIND PYTHON OOF PROJECTIONS
# =====================================================================

quant = pd.read_csv(OOF_MODEL_PATH)

quant = quant[
    [
        "game_id",
        "season",
        "pred_home_margin",
        "pred_total",
    ]
].drop_duplicates(
    subset=["game_id", "season"]
)

games = games.merge(
    quant,
    on=["game_id", "season"],
    how="inner",
    validate="one_to_one",
)


# =====================================================================
# ADD HISTORICAL OPEN/CLOSE MARKET
# =====================================================================

market = pd.read_csv(MARKET_PATH)

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
        "open_total",
        "close_total",
        "open_total_providers",
        "close_total_providers",
    ]
]

games = games.merge(
    market,
    on=["game_id", "season"],
    how="inner",
    validate="one_to_one",
)


# =====================================================================
# MARKET CONVENTIONS
# =====================================================================

games["open_home_margin"] = (
    -games["open_home_spread"]
)

games["close_home_margin"] = (
    -games["close_home_spread"]
)


# =====================================================================
# TARGETS = OPENING MARKET ERROR
# =====================================================================

games["spread_market_residual"] = (
    games["actual_home_margin"]
    - games["open_home_margin"]
)

games["total_market_residual"] = (
    games["actual_total"]
    - games["open_total"]
)


# =====================================================================
# PYTHON VS OPENING MARKET
# =====================================================================

games["quant_spread_edge"] = (
    games["pred_home_margin"]
    - games["open_home_margin"]
)

games["quant_total_edge"] = (
    games["pred_total"]
    - games["open_total"]
)

games["quant_spread_abs_edge"] = (
    games["quant_spread_edge"].abs()
)

games["quant_total_abs_edge"] = (
    games["quant_total_edge"].abs()
)


# =====================================================================
# FOOTBALL GAME FEATURES
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

    feature_data[f"diff__{col}"] = h - a
    feature_data[f"sum__{col}"] = h + a


X_football = pd.DataFrame(
    feature_data,
    index=games.index,
)

X_football = X_football.replace(
    [np.inf, -np.inf],
    np.nan,
)


# =====================================================================
# DECISION-TIME MARKET FEATURES
#
# NO CLOSES.
# NO FINAL SCORES.
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

X_market["pred_total"] = (
    games["pred_total"]
)

X_market["quant_spread_edge"] = (
    games["quant_spread_edge"]
)

X_market["quant_total_edge"] = (
    games["quant_total_edge"]
)

X_market["quant_spread_abs_edge"] = (
    games["quant_spread_abs_edge"]
)

X_market["quant_total_abs_edge"] = (
    games["quant_total_abs_edge"]
)


X = pd.concat(
    [X_football, X_market],
    axis=1,
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)


print(f"Paired market games: {len(games):,}")
print(f"Model features:      {X.shape[1]:,}")


# =====================================================================
# CHRONOLOGICAL OOF
# =====================================================================

models = [
    "ridge",
    "hist_gradient_boosting",
    "extra_trees",
]

targets = {
    "spread": "spread_market_residual",
    "total": "total_market_residual",
}

oof_rows = []


for market_name, target_col in targets.items():

    print("\n" + "=" * 120)
    print(f"TARGET: {market_name.upper()} MARKET RESIDUAL")
    print("=" * 120)

    for season in [
        2022,
        2023,
        2024,
        2025,
    ]:

        if market_name == "spread":
            has_market = games[
                "open_home_margin"
            ].notna()
        else:
            has_market = games[
                "open_total"
            ].notna()

        train_mask = (
            games["season"].lt(season)
            & games[target_col].notna()
            & has_market
        )

        test_mask = (
            games["season"].eq(season)
            & games[target_col].notna()
            & has_market
        )

        train_idx = games.index[train_mask]
        test_idx = games.index[test_mask]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        print(
            f"\n{season}: "
            f"train={len(train_idx):,} "
            f"test={len(test_idx):,}"
        )

        X_train = X.loc[train_idx]
        y_train = games.loc[
            train_idx,
            target_col,
        ]

        X_test = X.loc[test_idx]
        y_test = games.loc[
            test_idx,
            target_col,
        ]

        for model_name in models:

            print(
                f"  fitting {model_name:<25}",
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

            pred_residual = model.predict(
                X_test
            )

            frame = games.loc[
                test_idx,
                [
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "actual_home_margin",
                    "actual_total",
                    "open_home_margin",
                    "open_total",
                    "close_home_margin",
                    "close_total",
                ],
            ].copy()

            frame["market"] = market_name
            frame["model"] = model_name

            frame[
                "actual_market_residual"
            ] = y_test.to_numpy()

            frame[
                "pred_market_residual"
            ] = pred_residual

            if market_name == "spread":

                frame[
                    "market_baseline_prediction"
                ] = frame[
                    "open_home_margin"
                ]

                frame[
                    "residual_model_prediction"
                ] = (
                    frame[
                        "open_home_margin"
                    ]
                    + pred_residual
                )

                actual = frame[
                    "actual_home_margin"
                ].to_numpy()

            else:

                frame[
                    "market_baseline_prediction"
                ] = frame[
                    "open_total"
                ]

                frame[
                    "residual_model_prediction"
                ] = (
                    frame[
                        "open_total"
                    ]
                    + pred_residual
                )

                actual = frame[
                    "actual_total"
                ].to_numpy()

            baseline = frame[
                "market_baseline_prediction"
            ].to_numpy()

            corrected = frame[
                "residual_model_prediction"
            ].to_numpy()

            baseline_mae = np.mean(
                np.abs(
                    actual - baseline
                )
            )

            model_mae = np.mean(
                np.abs(
                    actual - corrected
                )
            )

            baseline_rmse = np.sqrt(
                np.mean(
                    (
                        actual - baseline
                    ) ** 2
                )
            )

            model_rmse = np.sqrt(
                np.mean(
                    (
                        actual - corrected
                    ) ** 2
                )
            )

            print(
                f" market MAE={baseline_mae:.3f}"
                f" model MAE={model_mae:.3f}"
                f" market RMSE={baseline_rmse:.3f}"
                f" model RMSE={model_rmse:.3f}"
            )

            oof_rows.append(
                frame
            )


# =====================================================================
# SAVE OOF
# =====================================================================

oof = pd.concat(
    oof_rows,
    ignore_index=True,
)

oof.to_parquet(
    OOF_OUT,
    index=False,
)


# =====================================================================
# YEAR-LEVEL SUMMARY
# =====================================================================

summary_rows = []

for (
    market_name,
    model_name,
    season,
), sub in oof.groupby(
    [
        "market",
        "model",
        "season",
    ]
):

    if market_name == "spread":
        actual = sub[
            "actual_home_margin"
        ]
    else:
        actual = sub[
            "actual_total"
        ]

    baseline_error = (
        actual
        - sub[
            "market_baseline_prediction"
        ]
    )

    model_error = (
        actual
        - sub[
            "residual_model_prediction"
        ]
    )

    summary_rows.append(
        {
            "market": market_name,
            "model": model_name,
            "season": season,
            "games": len(sub),

            "market_mae":
                baseline_error.abs().mean(),

            "residual_model_mae":
                model_error.abs().mean(),

            "mae_improvement":
                baseline_error.abs().mean()
                - model_error.abs().mean(),

            "market_rmse":
                np.sqrt(
                    np.mean(
                        baseline_error ** 2
                    )
                ),

            "residual_model_rmse":
                np.sqrt(
                    np.mean(
                        model_error ** 2
                    )
                ),

            "rmse_improvement":
                np.sqrt(
                    np.mean(
                        baseline_error ** 2
                    )
                )
                -
                np.sqrt(
                    np.mean(
                        model_error ** 2
                    )
                ),

            "residual_prediction_corr":
                sub[
                    "pred_market_residual"
                ].corr(
                    sub[
                        "actual_market_residual"
                    ]
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# =====================================================================
# DEVELOPMENT 2022-2024
# =====================================================================

print("\n" + "=" * 120)
print("DEVELOPMENT 2022-2024 AGGREGATE")
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

    if market_name == "spread":
        actual = sub[
            "actual_home_margin"
        ]
    else:
        actual = sub[
            "actual_total"
        ]

    baseline_error = (
        actual
        - sub[
            "market_baseline_prediction"
        ]
    )

    model_error = (
        actual
        - sub[
            "residual_model_prediction"
        ]
    )

    dev_rows.append(
        {
            "market":
                market_name,

            "model":
                model_name,

            "games":
                len(sub),

            "market_mae":
                baseline_error.abs().mean(),

            "model_mae":
                model_error.abs().mean(),

            "mae_gain":
                baseline_error.abs().mean()
                - model_error.abs().mean(),

            "market_rmse":
                np.sqrt(
                    np.mean(
                        baseline_error ** 2
                    )
                ),

            "model_rmse":
                np.sqrt(
                    np.mean(
                        model_error ** 2
                    )
                ),

            "rmse_gain":
                np.sqrt(
                    np.mean(
                        baseline_error ** 2
                    )
                )
                -
                np.sqrt(
                    np.mean(
                        model_error ** 2
                    )
                ),

            "resid_corr":
                sub[
                    "pred_market_residual"
                ].corr(
                    sub[
                        "actual_market_residual"
                    ]
                ),
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
# SELECT WINNERS USING DEVELOPMENT ONLY
# =====================================================================

winners = {}

for market_name in [
    "spread",
    "total",
]:

    x = dev[
        dev["market"].eq(
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
    ] = x.iloc[0]["model"]


print("\nDevelopment winners:")
print(winners)


# =====================================================================
# 2025 FIXED TEMPORAL TEST
# =====================================================================

print("\n" + "=" * 120)
print("2025 FIXED TEMPORAL TEST")
print("=" * 120)

hold_rows = []

for market_name, winner in winners.items():

    sub = oof[
        oof["season"].eq(2025)
        &
        oof["market"].eq(
            market_name
        )
        &
        oof["model"].eq(
            winner
        )
    ].copy()

    if market_name == "spread":
        actual = sub[
            "actual_home_margin"
        ]
    else:
        actual = sub[
            "actual_total"
        ]

    baseline_error = (
        actual
        - sub[
            "market_baseline_prediction"
        ]
    )

    model_error = (
        actual
        - sub[
            "residual_model_prediction"
        ]
    )

    hold_rows.append(
        {
            "market":
                market_name,

            "winner":
                winner,

            "games":
                len(sub),

            "market_mae":
                baseline_error.abs().mean(),

            "model_mae":
                model_error.abs().mean(),

            "mae_gain":
                baseline_error.abs().mean()
                - model_error.abs().mean(),

            "market_rmse":
                np.sqrt(
                    np.mean(
                        baseline_error ** 2
                    )
                ),

            "model_rmse":
                np.sqrt(
                    np.mean(
                        model_error ** 2
                    )
                ),

            "rmse_gain":
                np.sqrt(
                    np.mean(
                        baseline_error ** 2
                    )
                )
                -
                np.sqrt(
                    np.mean(
                        model_error ** 2
                    )
                ),

            "resid_corr":
                sub[
                    "pred_market_residual"
                ].corr(
                    sub[
                        "actual_market_residual"
                    ]
                ),
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
# WINNER YEAR-BY-YEAR
# =====================================================================

print("\n" + "=" * 120)
print("WINNER YEAR-BY-YEAR STABILITY")
print("=" * 120)

winner_year_rows = []

for market_name, winner in winners.items():

    x = summary[
        summary["market"].eq(
            market_name
        )
        &
        summary["model"].eq(
            winner
        )
    ].copy()

    winner_year_rows.append(
        x
    )

winner_year = pd.concat(
    winner_year_rows,
    ignore_index=True,
)

print(
    winner_year[
        [
            "market",
            "model",
            "season",
            "games",
            "market_mae",
            "residual_model_mae",
            "mae_improvement",
            "market_rmse",
            "residual_model_rmse",
            "rmse_improvement",
            "residual_prediction_corr",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("OUTPUTS")
print("=" * 120)

print("OOF:", OOF_OUT)
print("Summary:", SUMMARY_OUT)

print("=" * 120)
print("LEAKAGE GUARD:")
print("Closing lines and final scores are NOT model inputs.")
print("=" * 120)
