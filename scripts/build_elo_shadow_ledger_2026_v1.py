"""Step 37: Archived Ridge versus Elo shadow ledger.

Research only. No blending, retraining or production changes.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path.cwd()

RIDGE = ROOT / (
    "reports/locked_forecasts/"
    "20261029_preweek9_baseline/"
    "matchup_games_2026_week_9.csv"
)

ELO = ROOT / (
    "reports/research/elo_2026_spread_v1/"
    "elo_spread_forecasts.csv"
)

ELO_MANIFEST = ELO.parent / "manifest.json"

BRIDGE_MANIFEST = ROOT / (
    "reports/research/elo_2026_rating_bridge_v1/"
    "manifest.json"
)

FROZEN = ROOT / (
    "reports/research/elo_shadow_challenger_v1/"
    "frozen_model.json"
)

OUT = ROOT / (
    "reports/research/elo_shadow_challenger_v1/"
    "2026_shadow_ledger"
)

CSV = OUT / "paired_forecasts.csv"
MANIFEST = OUT / "manifest.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


require(
    all(path.is_file() for path in [
        RIDGE, ELO, ELO_MANIFEST,
        BRIDGE_MANIFEST, FROZEN
    ]),
    "Required forecast or provenance file missing"
)

require(
    not CSV.exists() and not MANIFEST.exists(),
    "Existing ledger detected. Refusing overwrite."
)

elo_info = json.loads(
    ELO_MANIFEST.read_text(encoding="utf-8")
)

bridge_info = json.loads(
    BRIDGE_MANIFEST.read_text(encoding="utf-8")
)

frozen = json.loads(
    FROZEN.read_text(encoding="utf-8")
)

require(
    sha256(ELO) == elo_info["forecast_sha256"],
    "Elo forecast SHA256 mismatch"
)

require(
    sha256(BRIDGE_MANIFEST)
    == elo_info["bridge_manifest_sha256"],
    "Bridge manifest SHA256 mismatch"
)

# Verify each historical hash against the file it identifies.
# The Elo features CSV and historical matchup Parquet are
# different artifacts and must not have identical hashes.

historical_elo = ROOT / (
    "reports/research/historical_elo_v1/"
    "pregame_elo_features.csv"
)

historical_matchups = ROOT / (
    "data/processed/cfb_matchup_features_opponent_v45.parquet"
)

bridge_ratings = ROOT / (
    "reports/research/elo_2026_rating_bridge_v1/"
    "pregame_ratings.csv"
)

require(
    sha256(historical_elo)
    == elo_info["historical_source_sha256"],
    "Historical Elo features SHA256 mismatch"
)

require(
    sha256(historical_matchups)
    == bridge_info["historical_source_sha256"],
    "Historical matchup Parquet SHA256 mismatch"
)

require(
    sha256(bridge_ratings)
    == bridge_info["output_sha256"],
    "Bridge ratings SHA256 mismatch"
)

require(
    sha256(bridge_ratings)
    == elo_info["bridge_ratings_sha256"],
    "Spread forecast references different bridge ratings"
)

require(
    elo_info["frozen_challenger_applied"] is False,
    "Unexpected pre-applied correction"
)

require(
    frozen["production_approved"] is False,
    "Unexpected frozen-model status"
)

ridge = pd.read_csv(
    RIDGE, dtype={"game_id": str}
)

elo = pd.read_csv(
    ELO, dtype={"game_id": str}
)

require(
    ridge.game_id.is_unique and elo.game_id.is_unique,
    "Duplicate forecast game IDs"
)

require(
    len(ridge) == 52,
    "Unexpected archived Ridge coverage"
)

require(
    len(elo) > 0,
    "No Elo forecasts available"
)

ridge = ridge[[
    "game_id",
    "start_date",
    "home_team",
    "away_team",
    "projected_home_margin",
    "home_points",
    "away_points",
    "projected_total",
]].copy()

ridge = ridge.rename(columns={
    "start_date": "ridge_kickoff",
    "projected_home_margin": "ridge_margin",
    "home_points": "ridge_home_points",
    "away_points": "ridge_away_points",
    "projected_total": "ridge_total",
})

elo = elo[[
    "game_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "elo_predicted_home_margin",
    "forecast_created_utc",
    "elo_difference",
]].copy()

elo = elo.rename(columns={
    "kickoff_utc": "elo_kickoff",
    "elo_predicted_home_margin": "elo_margin",
    "forecast_created_utc": "elo_created_utc",
})

joined = ridge.merge(
    elo,
    on="game_id",
    how="inner",
    suffixes=("_ridge", "_elo"),
    validate="one_to_one",
)

require(
    len(joined) == len(elo),
    "Some Elo forecasts have no archived Ridge match"
)

require(
    (
        joined.home_team_ridge
        == joined.home_team_elo
    ).all(),
    "Home-team identity mismatch"
)

require(
    (
        joined.away_team_ridge
        == joined.away_team_elo
    ).all(),
    "Away-team identity mismatch"
)

ridge_kickoff = pd.to_datetime(
    joined.ridge_kickoff,
    utc=True,
    errors="raise",
)

elo_kickoff = pd.to_datetime(
    joined.elo_kickoff,
    utc=True,
    errors="raise",
)

elo_created = pd.to_datetime(
    joined.elo_created_utc,
    utc=True,
    errors="raise",
)

require(
    ridge_kickoff.notna().all()
    and elo_kickoff.notna().all()
    and elo_created.notna().all(),
    "Missing timestamps"
)

require(
    (ridge_kickoff == elo_kickoff).all(),
    "Kickoff mismatch"
)

require(
    (elo_created < elo_kickoff).all(),
    "Elo forecast not created before kickoff"
)

require(
    (
        (
            joined.ridge_home_points
            - joined.ridge_away_points
        ) - joined.ridge_margin
    ).abs().le(0.002).all(),
    "Ridge margin arithmetic mismatch"
)

require(
    (
        (
            joined.ridge_home_points
            + joined.ridge_away_points
        ) - joined.ridge_total
    ).abs().le(0.002).all(),
    "Ridge total arithmetic mismatch"
)

numeric = [
    "ridge_margin",
    "elo_margin",
    "elo_difference",
]

require(
    np.isfinite(
        joined[numeric].to_numpy(dtype=float)
    ).all(),
    "Nonfinite forecast values"
)

created = datetime.now(timezone.utc)

joined["kickoff_utc"] = (
    elo_kickoff.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
)

joined["ridge_home_margin"] = joined.ridge_margin
joined["elo_home_margin"] = joined.elo_margin

joined["elo_minus_ridge"] = (
    joined.elo_margin - joined.ridge_margin
)

joined["forecast_status_at_ledger_creation"] = np.where(
    elo_kickoff > pd.Timestamp(created),
    "future",
    "kickoff_passed",
)

joined["ridge_model_compatibility_certified"] = False
joined["blended_forecast_generated"] = False
joined["production_approved"] = False

output_columns = [
    "game_id",
    "kickoff_utc",
    "away_team_ridge",
    "home_team_ridge",
    "ridge_home_margin",
    "elo_home_margin",
    "elo_minus_ridge",
    "ridge_total",
    "elo_created_utc",
    "forecast_status_at_ledger_creation",
    "ridge_model_compatibility_certified",
    "blended_forecast_generated",
    "production_approved",
]

result = joined[output_columns].rename(columns={
    "away_team_ridge": "away_team",
    "home_team_ridge": "home_team",
})

result = result.sort_values(
    ["kickoff_utc", "game_id"]
).reset_index(drop=True)

OUT.mkdir(parents=True, exist_ok=True)
result.to_csv(CSV, index=False)

manifest = {
    "schema": "elo_2026_shadow_ledger_v1",
    "created_at_utc": created.isoformat(),
    "ridge_source_sha256": sha256(RIDGE),
    "elo_source_sha256": sha256(ELO),
    "elo_manifest_sha256": sha256(ELO_MANIFEST),
    "bridge_manifest_sha256": sha256(BRIDGE_MANIFEST),
    "frozen_config_sha256": sha256(FROZEN),
    "output_sha256": sha256(CSV),
    "archived_ridge_games": len(ridge),
    "paired_forecasts": len(result),
    "future_at_ledger_creation": int(
        (
            result.forecast_status_at_ledger_creation
            == "future"
        ).sum()
    ),
    "ridge_model_compatibility_certified": False,
    "ridge_original_forecast_timestamp_certified": False,
    "elo_historical_pit_certified": False,
    "frozen_correction_applied": False,
    "production_approved": False,
    "research_only": True,
}

with MANIFEST.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print("\n=== STEP 37: SHADOW LEDGER COMPLETE ===")
print("Archived Ridge games:", len(ridge))
print("Matched Elo forecasts:", len(result))
print(
    "Future games at ledger creation:",
    manifest["future_at_ledger_creation"]
)
print("Team/kickoff validation: PASS")
print("Elo timestamp validation: PASS")
print("Ridge arithmetic validation: PASS")

print("\n=== FIRST TEN MATCHUPS ===")

print(
    result[[
        "away_team",
        "home_team",
        "ridge_home_margin",
        "elo_home_margin",
        "elo_minus_ridge",
    ]].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:+.2f}",
    )
)

print("\nFrozen blend applied: NO")
print("Ridge compatibility certified: NO")
print("Production modified: NO")
print("Output:", CSV.relative_to(ROOT))
print("SHA256:", sha256(CSV))
