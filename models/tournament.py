from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = "data/processed/cfb_pregame_features_v0_1.parquet"
DEFAULT_MANIFEST = "reports/cfb_feature_manifest_v0_1.json"
ARTIFACT_DIR = Path("models/artifacts")
REPORT_DIR = Path("reports")

TARGETS = [
    "pass_attempts",
    "completions",
    "passing_yards",
    "rush_attempts",
    "rushing_yards",
    "receptions",
    "receiving_yards",
]

TARGET_FAMILY = {
    "pass_attempts": "passing",
    "completions": "passing",
    "passing_yards": "passing",
    "rush_attempts": "rushing",
    "rushing_yards": "rushing",
    "receptions": "receiving",
    "receiving_yards": "receiving",
}

PREDICTION_BOUNDS = {
    "pass_attempts": (0.0, 80.0),
    "completions": (0.0, 60.0),
    "passing_yards": (0.0, 700.0),
    "rush_attempts": (0.0, 50.0),
    "rushing_yards": (-50.0, 450.0),
    "receptions": (0.0, 25.0),
    "receiving_yards": (-25.0, 400.0),
}

VALIDATION_YEARS = [2021, 2022, 2023, 2024]
HOLDOUT_YEAR = 2025
MIN_TRAIN_SEASON = 2014

# Pregame-only role-history eligibility. This deliberately does NOT use
# official_passer / official_rusher / official_receiver from the current game.
ELIGIBILITY_HISTORY = {
    "passing": [
        "p_pass_attempts_avg3",
        "p_pass_attempts_career_avg",
        "p_dropbacks_avg3",
        "p_dropbacks_career_avg",
    ],
    "rushing": [
        "p_rush_attempts_avg3",
        "p_rush_attempts_career_avg",
        "p_non_kneel_carries_avg3",
        "p_non_kneel_carries_career_avg",
    ],
    "receiving": [
        "p_receptions_avg3",
        "p_receptions_career_avg",
        "p_targets_pbp_avg3",
        "p_targets_pbp_career_avg",
    ],
}


def _to_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = frame.loc[:, columns].copy()
    for col in columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def _target_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    family = TARGET_FAMILY[target]
    mask = pd.Series(True, index=frame.index)

    if "player_games_before" in frame.columns:
        mask &= pd.to_numeric(
            frame["player_games_before"], errors="coerce"
        ).fillna(0).ge(1)

    history_cols = [
        c for c in ELIGIBILITY_HISTORY[family] if c in frame.columns
    ]
    if not history_cols:
        raise RuntimeError(
            f"No pregame eligibility history columns available for {family}."
        )

    history_signal = pd.Series(False, index=frame.index)
    for col in history_cols:
        history_signal |= pd.to_numeric(
            frame[col], errors="coerce"
        ).fillna(0).gt(0.25)

    mask &= history_signal
    mask &= pd.to_numeric(frame[target], errors="coerce").notna()

    # Team pseudo-rows are never betting candidates. is_team_row is used only
    # as a sample-quality exclusion and is not passed to the models.
    if "is_team_row" in frame.columns:
        team_row = frame["is_team_row"].fillna(False).astype(bool)
        mask &= ~team_row

    return mask


class RecentHistoryBaseline(BaseEstimator, RegressorMixin):
    """Pregame baseline using shifted recent/season/career history only."""

    def __init__(self, target: str):
        self.target = target
        self.global_mean_: float | None = None
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = float(pd.to_numeric(y, errors="coerce").mean())
        candidates = [
            f"p_{self.target}_avg3",
            f"p_{self.target}_avg5",
            f"p_{self.target}_season_avg",
            f"p_{self.target}_career_avg",
            f"p_{self.target}_lag1",
        ]
        self.columns_ = [c for c in candidates if c in X.columns]
        if not self.columns_:
            raise RuntimeError(
                f"No historical baseline columns found for {self.target}."
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("RecentHistoryBaseline has not been fit.")
        out = pd.Series(np.nan, index=X.index, dtype=float)
        for col in self.columns_:
            values = pd.to_numeric(X[col], errors="coerce")
            out = out.fillna(values)
        out = out.fillna(self.global_mean_)
        return out.to_numpy(dtype=float)


def _make_ridge(target: str):
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=20.0)),
        ]
    )


def _make_hgb(target: str):
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def _make_extra_trees(target: str):
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=120,
                    max_depth=18,
                    min_samples_leaf=4,
                    max_features=0.70,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


MODEL_FACTORIES: dict[str, Callable[[str], BaseEstimator]] = {
    "recent_baseline": lambda target: RecentHistoryBaseline(target),
    "ridge": _make_ridge,
    "hist_gradient_boosting": _make_hgb,
    "extra_trees": _make_extra_trees,
}


def _clip_predictions(target: str, predictions: np.ndarray) -> np.ndarray:
    lo, hi = PREDICTION_BOUNDS[target]
    return np.clip(np.asarray(predictions, dtype=float), lo, hi)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(math.sqrt(mean_squared_error(actual, predicted)))
    bias = float(np.mean(predicted - actual))
    if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0:
        corr = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
    }


def _load_inputs(
    data_path: str,
    manifest_path: str,
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(data_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    feature_columns = list(manifest["feature_columns"])

    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Feature table is missing {len(missing)} manifest columns: "
            f"{missing[:20]}"
        )

    direct_leaks = sorted(set(TARGETS).intersection(feature_columns))
    if direct_leaks:
        raise RuntimeError(f"Current-game targets leaked into features: {direct_leaks}")

    if HOLDOUT_YEAR not in set(pd.to_numeric(frame["season"], errors="coerce")):
        raise RuntimeError(f"{HOLDOUT_YEAR} holdout season not present.")

    return frame, feature_columns


def _fit_and_predict(
    model_name: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[BaseEstimator, np.ndarray]:
    model = MODEL_FACTORIES[model_name](target)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return model, _clip_predictions(target, pred)


def run_validation(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []

    for target in TARGETS:
        eligible = _target_mask(frame, target)

        for validation_year in VALIDATION_YEARS:
            train_mask = (
                eligible
                & frame["season"].between(MIN_TRAIN_SEASON, validation_year - 1)
            )
            valid_mask = eligible & frame["season"].eq(validation_year)

            train = frame.loc[train_mask]
            valid = frame.loc[valid_mask]

            if train.empty or valid.empty:
                raise RuntimeError(
                    f"Empty split for {target}, validation {validation_year}: "
                    f"train={len(train)}, valid={len(valid)}"
                )

            # Hard holdout guard.
            if int(train["season"].max()) >= validation_year:
                raise RuntimeError("Chronological leakage detected in validation split.")
            if HOLDOUT_YEAR in set(train["season"]) or HOLDOUT_YEAR in set(valid["season"]):
                raise RuntimeError("2025 holdout entered model-selection validation.")

            X_train = _to_numeric_frame(train, feature_columns)
            y_train = pd.to_numeric(train[target], errors="coerce")
            X_valid = _to_numeric_frame(valid, feature_columns)
            y_valid = pd.to_numeric(valid[target], errors="coerce").to_numpy(dtype=float)

            for model_name in MODEL_FACTORIES:
                _, pred = _fit_and_predict(
                    model_name,
                    target,
                    X_train,
                    y_train,
                    X_valid,
                )
                metric = _metrics(y_valid, pred)
                rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "validation_year": validation_year,
                        "train_start_season": int(train["season"].min()),
                        "train_end_season": int(train["season"].max()),
                        "train_rows": int(len(train)),
                        "validation_rows": int(len(valid)),
                        **metric,
                    }
                )
                print(
                    f"{target:18s} | {validation_year} | "
                    f"{model_name:22s} | MAE {metric['mae']:.4f}"
                )

    return pd.DataFrame(rows)


def summarize_validation(validation: pd.DataFrame) -> pd.DataFrame:
    summary = (
        validation.groupby(["target", "model"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_abs_bias=("bias", lambda s: float(np.mean(np.abs(s)))),
            mean_corr=("corr", "mean"),
            folds=("validation_year", "nunique"),
            total_validation_rows=("validation_rows", "sum"),
        )
    )

    baseline = (
        summary.loc[
            summary["model"].eq("recent_baseline"),
            ["target", "mean_mae"],
        ]
        .rename(columns={"mean_mae": "baseline_mean_mae"})
    )
    summary = summary.merge(baseline, on="target", how="left")
    summary["mae_improvement_vs_baseline_pct"] = (
        100.0
        * (summary["baseline_mean_mae"] - summary["mean_mae"])
        / summary["baseline_mean_mae"]
    )
    return summary.sort_values(
        ["target", "mean_mae", "mean_rmse", "mean_abs_bias"],
        kind="mergesort",
    ).reset_index(drop=True)


def choose_winners(summary: pd.DataFrame) -> dict[str, str]:
    winners: dict[str, str] = {}
    for target in TARGETS:
        candidates = summary.loc[summary["target"].eq(target)].sort_values(
            ["mean_mae", "mean_rmse", "mean_abs_bias", "model"],
            kind="mergesort",
        )
        if candidates.empty:
            raise RuntimeError(f"No validation results for {target}.")
        winners[target] = str(candidates.iloc[0]["model"])
    return winners


def fit_final_and_holdout(
    frame: pd.DataFrame,
    feature_columns: list[str],
    winners: dict[str, str],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for target in TARGETS:
        eligible = _target_mask(frame, target)
        train_mask = (
            eligible
            & frame["season"].between(MIN_TRAIN_SEASON, HOLDOUT_YEAR - 1)
        )
        holdout_mask = eligible & frame["season"].eq(HOLDOUT_YEAR)

        train = frame.loc[train_mask]
        holdout = frame.loc[holdout_mask]

        if train.empty or holdout.empty:
            raise RuntimeError(
                f"Empty final split for {target}: "
                f"train={len(train)}, holdout={len(holdout)}"
            )
        if int(train["season"].max()) != HOLDOUT_YEAR - 1:
            raise RuntimeError(
                f"{target}: final training must stop at {HOLDOUT_YEAR - 1}."
            )
        if HOLDOUT_YEAR in set(train["season"]):
            raise RuntimeError("Holdout leakage into final training.")

        model_name = winners[target]

        X_train = _to_numeric_frame(train, feature_columns)
        y_train = pd.to_numeric(train[target], errors="coerce")
        X_holdout = _to_numeric_frame(holdout, feature_columns)
        y_holdout = pd.to_numeric(
            holdout[target], errors="coerce"
        ).to_numpy(dtype=float)

        model, pred = _fit_and_predict(
            model_name,
            target,
            X_train,
            y_train,
            X_holdout,
        )
        metric = _metrics(y_holdout, pred)

        winner_validation = summary.loc[
            (summary["target"].eq(target))
            & (summary["model"].eq(model_name))
        ].iloc[0]

        artifact = {
            "artifact_version": "v0.1",
            "target": target,
            "model_name": model_name,
            "model": model,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "training_rows": int(len(train)),
            "training_seasons": sorted(int(x) for x in train["season"].unique()),
            "max_train_season": int(train["season"].max()),
            "eligibility_family": TARGET_FAMILY[target],
            "eligibility_history_columns": [
                c
                for c in ELIGIBILITY_HISTORY[TARGET_FAMILY[target]]
                if c in frame.columns
            ],
            "prediction_bounds": PREDICTION_BOUNDS[target],
            "validation_mean_mae": float(winner_validation["mean_mae"]),
            "validation_mean_rmse": float(winner_validation["mean_rmse"]),
            "holdout_year": HOLDOUT_YEAR,
            "holdout_rows": int(len(holdout)),
            "holdout_mae": metric["mae"],
            "holdout_rmse": metric["rmse"],
            "holdout_bias": metric["bias"],
            "holdout_corr": metric["corr"],
            "holdout_used_for_model_selection": False,
        }

        artifact_path = ARTIFACT_DIR / f"cfb_{target}_quant_v0_1.joblib"
        joblib.dump(artifact, artifact_path)

        rows.append(
            {
                "target": target,
                "winner": model_name,
                "train_rows": int(len(train)),
                "holdout_rows": int(len(holdout)),
                **metric,
                "artifact": str(artifact_path),
            }
        )

        print(
            f"HOLDOUT {target:18s} | {model_name:22s} | "
            f"MAE {metric['mae']:.4f} | RMSE {metric['rmse']:.4f}"
        )

    return pd.DataFrame(rows)


def _write_markdown(
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    holdout: pd.DataFrame,
    winners: dict[str, str],
    feature_count: int,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "model_tournament_v0_1.md"

    lines = [
        "# CFB Model Tournament v0.1",
        "",
        "## Protocol",
        "",
        f"- Feature count: {feature_count}",
        "- Model selection uses chronological walk-forward validation only.",
        "- Validation folds:",
        "  - train 2014-2020 -> validate 2021",
        "  - train 2014-2021 -> validate 2022",
        "  - train 2014-2022 -> validate 2023",
        "  - train 2014-2023 -> validate 2024",
        "- 2025 is excluded from all model selection.",
        "- After winners are selected, each winner is refit on 2014-2024 and evaluated once on 2025.",
        "- Eligibility uses pregame historical role signals only; current-game role flags are not model features or eligibility inputs.",
        "",
        "## Winners from 2021-2024 validation",
        "",
        "| Target | Winner | Validation MAE | Improvement vs recent baseline |",
        "|---|---|---:|---:|",
    ]

    for target in TARGETS:
        winner = winners[target]
        row = summary.loc[
            (summary["target"].eq(target)) & (summary["model"].eq(winner))
        ].iloc[0]
        lines.append(
            f"| {target} | {winner} | {row['mean_mae']:.4f} | "
            f"{row['mae_improvement_vs_baseline_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 2025 untouched holdout",
        "",
        "| Target | Winner | Rows | MAE | RMSE | Bias | Corr |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in holdout.iterrows():
        corr = row["corr"]
        corr_text = "nan" if pd.isna(corr) else f"{corr:.4f}"
        lines.append(
            f"| {row['target']} | {row['winner']} | {int(row['holdout_rows'])} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['bias']:.4f} | {corr_text} |"
        )

    lines += [
        "",
        "## Controls",
        "",
        "- No random train/test split.",
        "- 2025 is not used for model or hyperparameter selection.",
        "- Current-game target columns are not in the manifest feature list.",
        "- Predictions are clipped only to fixed, predeclared target bounds.",
        "- Tournament results establish predictive validation only; they do not establish sportsbook profitability.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chronological CFB player-prop model tournament."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CFB MODEL TOURNAMENT v0.1")
    print("=" * 80)

    frame, feature_columns = _load_inputs(args.data, args.manifest)
    print(f"Rows loaded: {len(frame):,}")
    print(f"Features: {len(feature_columns):,}")
    print(f"Seasons: {int(frame['season'].min())}-{int(frame['season'].max())}")
    print()

    validation = run_validation(frame, feature_columns)
    validation.to_csv(
        REPORT_DIR / "model_tournament_validation_v0_1.csv",
        index=False,
    )

    summary = summarize_validation(validation)
    summary.to_csv(
        REPORT_DIR / "model_tournament_summary_v0_1.csv",
        index=False,
    )

    winners = choose_winners(summary)

    print()
    print("VALIDATION WINNERS")
    print("-" * 80)
    for target in TARGETS:
        print(f"{target:18s}: {winners[target]}")

    print()
    print("FINAL 2025 HOLDOUT")
    print("-" * 80)
    holdout = fit_final_and_holdout(
        frame,
        feature_columns,
        winners,
        summary,
    )
    holdout.to_csv(
        REPORT_DIR / "model_tournament_holdout_2025_v0_1.csv",
        index=False,
    )

    _write_markdown(
        validation,
        summary,
        holdout,
        winners,
        len(feature_columns),
    )

    print()
    print("=" * 80)
    print("CFB MODEL TOURNAMENT COMPLETE")
    print("=" * 80)
    print("Reports:")
    print("  reports/model_tournament_validation_v0_1.csv")
    print("  reports/model_tournament_summary_v0_1.csv")
    print("  reports/model_tournament_holdout_2025_v0_1.csv")
    print("  reports/model_tournament_v0_1.md")
    print("Artifacts:")
    for target in TARGETS:
        print(f"  models/artifacts/cfb_{target}_quant_v0_1.joblib")


if __name__ == "__main__":
    main()