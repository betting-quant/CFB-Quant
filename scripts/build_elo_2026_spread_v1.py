"""2026 Elo-only spread calibration and forecast.
Research-only. No Ridge blending or production modifications.
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path.cwd()

HIST = ROOT / (
    "reports/research/historical_elo_v1/"
    "pregame_elo_features.csv"
)
BRIDGE = ROOT / (
    "reports/research/elo_2026_rating_bridge_v1/"
    "pregame_ratings.csv"
)
BRIDGE_MANIFEST = BRIDGE.parent / "manifest.json"

OUT = ROOT / "reports/research/elo_2026_spread_v1"
CSV = OUT / "elo_spread_forecasts.csv"
MANIFEST = OUT / "manifest.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


require(
    HIST.is_file()
    and BRIDGE.is_file()
    and BRIDGE_MANIFEST.is_file(),
    "Required historical or bridge artifact missing"
)

require(
    not CSV.exists() and not MANIFEST.exists(),
    "Forecast artifact already exists; refusing overwrite"
)

bridge_info = json.loads(
    BRIDGE_MANIFEST.read_text(encoding="utf-8")
)

require(
    sha256(BRIDGE) == bridge_info["output_sha256"],
    "Bridge output integrity failure"
)
require(
    bridge_info["research_only"] is True
    and bridge_info["production_approved"] is False,
    "Unexpected bridge approval status"
)
require(
    bridge_info["historical_point_in_time_certified"] is False,
    "Unexpected historical certification status"
)

hist = pd.read_csv(HIST, dtype={"game_id": str})
future = pd.read_csv(BRIDGE, dtype={"game_id": str})

require(
    hist.game_id.is_unique and future.game_id.is_unique,
    "Duplicate historical or future game IDs"
)

require(
    set(hist.season.unique()) == set(range(2014, 2026)),
    "Unexpected training seasons"
)

require(
    set(hist.game_id).isdisjoint(set(future.game_id)),
    "Historical and future game IDs overlap"
)

X = hist[["elo_rating_difference"]].to_numpy(
    dtype=float
)
y = hist["actual_home_margin"].to_numpy(
    dtype=float
)

require(
    np.isfinite(X).all() and np.isfinite(y).all(),
    "Nonfinite calibration inputs"
)

require(
    len(hist) == 10369,
    "Historical training coverage changed"
)

model = LinearRegression()
model.fit(X, y)

future["kickoff_utc"] = pd.to_datetime(
    future["kickoff_utc"],
    utc=True,
    errors="raise"
)

require(
    future.kickoff_utc.notna().all(),
    "Missing future kickoff timestamps"
)

created = datetime.now(timezone.utc)

# A forecast may only be newly issued for a future game.
eligible = future.loc[
    future.kickoff_utc >
    created + timedelta(minutes=10)
].copy()

expired = future.loc[
    future.kickoff_utc <=
    created + timedelta(minutes=10),
    "game_id"
].tolist()

require(
    len(eligible) > 0,
    "No games remain eligible for new pregame forecasts"
)

require(
    len(eligible) + len(expired) == len(future),
    "Future game coverage accounting mismatch"
)

x_future = eligible[["elo_difference"]].to_numpy(
    dtype=float
)

require(
    np.isfinite(x_future).all(),
    "Invalid 2026 Elo rating differences"
)

eligible["elo_predicted_home_margin"] = model.predict(
    x_future
)

require(
    np.isfinite(
        eligible["elo_predicted_home_margin"].to_numpy()
    ).all(),
    "Nonfinite forecast margins"
)

eligible["forecast_created_utc"] = created.isoformat()
eligible["calibration_train_max_season"] = 2025
eligible["research_only"] = True

eligible = eligible.sort_values(
    ["kickoff_utc", "game_id"]
).reset_index(drop=True)

OUT.mkdir(parents=True, exist_ok=True)
eligible.to_csv(CSV, index=False)

manifest = {
    "schema": "elo_2026_spread_research_v1",
    "created_at_utc": created.isoformat(),
    "historical_source_sha256": sha256(HIST),
    "bridge_manifest_sha256": sha256(BRIDGE_MANIFEST),
    "bridge_ratings_sha256": sha256(BRIDGE),
    "forecast_sha256": sha256(CSV),
    "model": "linear_regression_elo_difference_only",
    "training_seasons": list(range(2014, 2026)),
    "training_games": len(hist),
    "intercept": float(model.intercept_),
    "elo_coefficient": float(model.coef_[0]),
    "margin_convention": "home_minus_away",
    "forecast_games": len(eligible),
    "expired_since_bridge": expired,
    "forecast_timestamp_policy": "created_before_kickoff",
    "historical_result_timing_certified": False,
    "2026_weekly_result_timestamps_verified": False,
    "frozen_challenger_applied": False,
    "ridge_baseline_compatibility_verified": False,
    "production_approved": False,
    "research_only": True,
}

with MANIFEST.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print("\n=== STEP 35: 2026 ELO SPREAD CALIBRATION ===")
print("Training seasons: 2014–2025")
print("Training games:", len(hist))
print("Intercept:", round(float(model.intercept_), 6))
print("Elo coefficient:", round(float(model.coef_[0]), 6))

print("\n=== 2026 FORECAST COVERAGE ===")
print("Bridge games:", len(future))
print("Forecast games:", len(eligible))
print("Expired since bridge:", len(expired))

print("\n=== FIRST TEN ELO FORECASTS ===")
print(
    eligible[[
        "away_team",
        "home_team",
        "elo_difference",
        "elo_predicted_home_margin"
    ]].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nOutput:", CSV.relative_to(ROOT))
print("Forecast SHA256:", sha256(CSV))
print("Frozen challenger applied: NO")
print("Production modified: NO")
