"""Prospective v5.8 fixed-blend research forecaster.

Uses the production v5.7 forecast pipeline without modifying it.
Requires a saved, eligible, pregame v5.6 feature snapshot.
Never permits archived/started games for prospective evaluation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cfb_quant.live import predict_v57 as live

ROOT_REPORTS = Path("reports/v58_development")
FEATURE_FILE = ROOT_REPORTS / "feature_dataset_v2/cfb_matchup_features_v58_research.parquet"
MANIFEST_FILE = ROOT_REPORTS / "feature_dataset_v2/feature_manifest.json"
HIST_IDS = ROOT_REPORTS / "historical_paired_predictions.csv"
BASE2025 = ROOT_REPORTS / "2025_phase_calibration_v58/game_predictions.csv"
OUTPUT = Path("reports/v58_prospective_blend")

WEIGHT = 0.15
SEED = 20261020

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm(series):
    return (series.astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True))

def prepare_training(root):
    path = root / FEATURE_FILE
    features = json.loads(
        (root / MANIFEST_FILE).read_text(encoding="utf-8")
    )["feature_names"]

    # The live v3 schema is the first 267 columns in v4.5.
    live_features = live.load_manifest(root, live.V3_MANIFEST)

    if len(features) != 267 or features != live_features:
        raise ValueError("Historical/live v3 feature schema differs")

    frame = pd.read_parquet(path)
    frame = frame.loc[frame["season"].between(2014, 2025)].copy()
    frame["game_id"] = norm(frame["game_id"])
    frame["home_away"] = frame["home_away"].astype(str).str.lower()

    keys = ["season", "game_id"]

    if frame.duplicated(keys + ["home_away"]).any():
        raise ValueError("Duplicate historical game-side records")

    home = frame.loc[frame.home_away.eq("home")].set_index(keys).sort_index()
    away = frame.loc[frame.home_away.eq("away")].set_index(keys).sort_index()

    if not home.index.equals(away.index):
        raise ValueError("Unpaired historical games")

    if not np.array_equal(
        home["opponent"].astype(str).to_numpy(),
        away["team"].astype(str).to_numpy()
    ):
        raise ValueError("Historical home/away identity mismatch")

    if not np.array_equal(
        away["opponent"].astype(str).to_numpy(),
        home["team"].astype(str).to_numpy()
    ):
        raise ValueError("Historical away/home identity mismatch")

    X = home[features].apply(pd.to_numeric, errors="raise").subtract(
        away[features].apply(pd.to_numeric, errors="raise")
    )
    X = X.replace([np.inf, -np.inf], np.nan)
    X.columns = ["diff_" + c for c in features]

    targets = (
        pd.to_numeric(home["points"], errors="raise")
        - pd.to_numeric(away["points"], errors="raise")
    )

    games = X.reset_index()
    games["target"] = targets.to_numpy(float)

    hist = pd.read_csv(root / HIST_IDS, usecols=keys)
    y2025 = pd.read_csv(root / BASE2025, usecols=keys)

    hist["game_id"] = norm(hist["game_id"])
    y2025["game_id"] = norm(y2025["game_id"])

    if hist.duplicated(keys).any() or y2025.duplicated(keys).any():
        raise ValueError("Duplicate eligibility IDs")

    if not hist.season.between(2019, 2024).all():
        raise ValueError("Invalid historical eligibility seasons")
    if not y2025.season.eq(2025).all():
        raise ValueError("Invalid 2025 eligibility season")

    hist_ids = pd.MultiIndex.from_frame(hist[keys])
    y2025_ids = pd.MultiIndex.from_frame(y2025[keys])
    game_ids = pd.MultiIndex.from_frame(games[keys])

    if not hist_ids.isin(game_ids).all():
        raise ValueError("Missing historical eligible games")
    if not y2025_ids.isin(game_ids).all():
        raise ValueError("Missing 2025 eligible games")

    eligible = (
        games.season.lt(2019)
        | (games.season.between(2019, 2024) & game_ids.isin(hist_ids))
        | (games.season.eq(2025) & game_ids.isin(y2025_ids))
    )

    train = games.loc[eligible].copy()

    if train.season.max() != 2025:
        raise ValueError("Unexpected training cutoff")
    if not np.isfinite(train.target.to_numpy(float)).all():
        raise ValueError("Invalid training targets")

    return train, features, list(X.columns)

def prepare_live(snapshot_frame, forecast, features, diff_cols):
    future = snapshot_frame.copy()
    future["game_id"] = norm(future["game_id"])
    forecast = forecast.copy()
    forecast["game_id"] = norm(forecast["game_id"])

    # Work only with the precise games the production run forecast.
    future = future.loc[
        future.game_id.isin(set(forecast.game_id))
    ].copy()

    if future.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Duplicate live game-side records")

    home = future.loc[
        future.home_away.eq("home")
    ].set_index("game_id").sort_index()
    away = future.loc[
        future.home_away.eq("away")
    ].set_index("game_id").sort_index()

    if not home.index.equals(away.index):
        raise ValueError("Incomplete live pairs")

    if not np.array_equal(
        home["opponent"].astype(str).to_numpy(),
        away["team"].astype(str).to_numpy()
    ):
        raise ValueError("Live opponent identities disagree")

    if not np.array_equal(
        away["opponent"].astype(str).to_numpy(),
        home["team"].astype(str).to_numpy()
    ):
        raise ValueError("Live away opponent identities disagree")

    X = home[features].apply(pd.to_numeric, errors="raise").subtract(
        away[features].apply(pd.to_numeric, errors="raise")
    )
    X = X.replace([np.inf, -np.inf], np.nan)
    X.columns = diff_cols

    if len(X) != len(forecast):
        raise ValueError("Live feature/forecast population mismatch")

    if set(X.index) != set(forecast.game_id):
        raise ValueError("Live game IDs mismatch")

    return X.reindex(forecast.game_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print("STEP 122 — PROSPECTIVE BLEND", flush=True)

    # Reuse the complete production validation and forecasting process.
    production, production_audit = live.run(
        root, Path(args.snapshot),
        samples=10000,
        seed=57,
        allow_started=False
    )

    print("Production predictions:", len(production), flush=True)

    snapshot_dir = Path(
        production_audit["snapshot_validation"]["snapshot_folder"]
    )
    snapshot_file = snapshot_dir / "pregame_features.parquet"
    snapshot_rows = pd.read_parquet(snapshot_file)

    train, features, diff_cols = prepare_training(root)
    X_live = prepare_live(
        snapshot_rows, production, features, diff_cols
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=125,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=20.0,
        max_bins=128,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(train[diff_cols], train["target"])

    challenger = model.predict(X_live)
    if not np.isfinite(challenger).all():
        raise ValueError("Nonfinite challenger predictions")

    result = production[[
        "game_id", "kickoff_utc", "home", "away",
        "fair_margin_home", "fair_total"
    ]].copy()

    result["boosting_margin_home"] = challenger
    result["blend_margin_home"] = (
        (1 - WEIGHT) * result["fair_margin_home"]
        + WEIGHT * result["boosting_margin_home"]
    )
    result["blend_home_spread"] = -result["blend_margin_home"]

    # Fail closed if model fitting ran into the start of any game.
    completed_at = pd.Timestamp(datetime.now(timezone.utc))
    kicks = pd.to_datetime(result.kickoff_utc, utc=True)

    if (kicks <= completed_at + pd.Timedelta(seconds=60)).any():
        raise RuntimeError(
            "At least one kickoff is too close or already started. "
            "Refusing prospective archive."
        )

    if result.game_id.duplicated().any():
        raise ValueError("Duplicate output game IDs")

    output_root = root / OUTPUT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_root / stamp

    # Save only after all forecast and kickoff checks pass.
    destination.mkdir(parents=True, exist_ok=False)

    prediction_file = destination / "paired_forecasts.csv"
    result.to_csv(prediction_file, index=False)

    inputs = {
        str(p): sha(root / p)
        for p in [
            FEATURE_FILE, MANIFEST_FILE,
            HIST_IDS, BASE2025
        ]
    }

    audit = {
        "version": "v5.8 prospective blend research",
        "generated_at_utc": completed_at.isoformat(),
        "production_generated_at_utc": production_audit["generated_at_utc"],
        "games": len(result),
        "earliest_kickoff_utc": kicks.min().isoformat(),
        "blend_weight": WEIGHT,
        "boosting_training_rows": len(train),
        "boosting_training_last_season": int(train.season.max()),
        "boosting_random_seed": SEED,
        "boosting_config": model.get_params(),
        "input_sha256": inputs,
        "snapshot_sha256": sha(snapshot_file),
        "snapshot_audit_sha256": sha(snapshot_dir / "audit.json"),
        "production_audit": production_audit,
        "paired_forecasts_sha256": sha(prediction_file),
        "research_only": True,
        "promotion_eligible": False,
        "disclaimer": (
            "Local timestamp and hashes support auditability, "
            "but are not independent immutable proof of pregame creation. "
            "Historical feature point-in-time provenance remains incomplete."
        ),
    }

    audit_path = destination / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, default=str),
        encoding="utf-8"
    )

    print("\nPROSPECTIVE FORECASTS")
    print(result.round(3).to_string(index=False))
    print("\nARCHIVE:", destination)
    print("PREDICTIONS SHA256:", audit["paired_forecasts_sha256"])
    print("Production v5.7: UNCHANGED")

if __name__ == "__main__":
    main()
