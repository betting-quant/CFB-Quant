"""Step 73: Full kickoff-ordered historical research rebuild."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from cfb_quant.features.matchup_engine import build_features

ROOT = Path.cwd()
OUT = ROOT / "reports/research/full_chronology_rebuild_v2"

FULL = ROOT / (
    "reports/research/full_2026_source_recovery_v1"
)
RAW_2026 = FULL / "team_games_2026_complete.parquet"
MAP_2026 = FULL / "team_game_kickoff_map.csv"
MANIFEST_2026 = FULL / "manifest.json"

OLD_MAP = ROOT / (
    "reports/spread_total_schedule_chronology_v2/"
    "team_game_kickoff_map.csv"
)
ORIGINAL = ROOT / (
    "data/processed/cfb_matchup_features_20261029.parquet"
)
FEATURE_MANIFEST = ROOT / (
    "reports/cfb_matchup_feature_manifest_20261029.json"
)

KEYS = ["game_id", "season", "team", "home_away"]

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize(frame):
    df = frame.copy()
    df["game_id"] = (
        df.game_id.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    df["season"] = pd.to_numeric(
        df.season, errors="raise"
    ).astype(int)
    df["team"] = df.team.astype(str).str.strip()
    df["home_away"] = (
        df.home_away.astype(str).str.strip().str.lower()
    )
    return df

def same(a, b):
    return np.isclose(
        pd.to_numeric(a, errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(b, errors="coerce").to_numpy(dtype=float),
        atol=1e-8,
        rtol=1e-9,
        equal_nan=True
    )

print("\n=== STEP 73: FULL HISTORICAL REBUILD ===")

inputs = [
    RAW_2026, MAP_2026, MANIFEST_2026,
    OLD_MAP, ORIGINAL, FEATURE_MANIFEST
]

require(
    all(path.is_file() for path in inputs),
    "Required source or manifest missing"
)
require(
    not OUT.exists(),
    "Output already exists; refusing overwrite"
)

recovery_manifest = json.loads(
    MANIFEST_2026.read_text(encoding="utf-8-sig")
)

require(
    sha(RAW_2026) ==
    recovery_manifest["outputs_sha256"][RAW_2026.name],
    "Recovered raw source SHA256 mismatch"
)
require(
    sha(MAP_2026) ==
    recovery_manifest["outputs_sha256"][MAP_2026.name],
    "Recovered kickoff map SHA256 mismatch"
)

feature_manifest = json.loads(
    FEATURE_MANIFEST.read_text(encoding="utf-8-sig")
)
features = feature_manifest["feature_columns"]

require(
    len(features) == 267 and len(set(features)) == 267,
    "Original 267-feature contract changed"
)

historical_paths = [
    ROOT / f"data/processed/team_games_{year}.parquet"
    for year in range(2014, 2026)
]

require(
    all(p.is_file() for p in historical_paths),
    "Missing 2014-2025 raw history"
)

historical = pd.concat(
    [pd.read_parquet(p) for p in historical_paths],
    ignore_index=True
)
historical = normalize(historical)

current_2026 = normalize(pd.read_parquet(
    ROOT / "data/processed/team_games_2026.parquet"
))
recovered_2026 = normalize(pd.read_parquet(RAW_2026))

require(
    len(historical) == 20738,
    "Historical row count mismatch"
)
require(
    len(current_2026) == 910,
    "Existing 2026 source count changed"
)
require(
    len(recovered_2026) == 1140,
    "Recovered 2026 source count mismatch"
)

require(
    set(historical.columns) == set(recovered_2026.columns),
    "Historical/recovered schema mismatch"
)

# Ensure the recovered dataset preserves all existing raw values.
paired_raw = current_2026.merge(
    recovered_2026,
    on=KEYS,
    how="inner",
    suffixes=("_old", "_new"),
    validate="one_to_one"
)
require(
    len(paired_raw) == 910,
    "Recovered source lost original raw identities"
)

for col in current_2026.columns:
    if col in KEYS:
        continue
    old = paired_raw[f"{col}_old"]
    new = paired_raw[f"{col}_new"]

    if pd.api.types.is_numeric_dtype(current_2026[col]):
        identical = same(old, new).all()
    else:
        identical = (
            old.astype("string").eq(new.astype("string"))
            .fillna(False)
            | (old.isna() & new.isna())
        ).all()

    require(bool(identical), f"Raw source changed: {col}")

raw = pd.concat(
    [
        historical,
        recovered_2026[historical.columns]
    ],
    ignore_index=True
)

require(
    len(raw) == 21878
    and not raw.duplicated(KEYS).any(),
    "Combined raw identity failure"
)
require(
    raw["points"].notna().all()
    and raw["opponent_points"].notna().all(),
    "Incomplete source scores"
)

print("Raw team-game rows:", len(raw))
print("Raw games:", raw.game_id.nunique())

# Build the full kickoff mapping.
old_times = normalize(pd.read_csv(OLD_MAP))
new_times = normalize(pd.read_csv(MAP_2026))

old_times = old_times.loc[
    old_times.season.lt(2026),
    KEYS + ["kickoff_utc"]
].copy()

new_times = new_times[
    KEYS + ["kickoff_utc"]
].copy()

# Parse independently: historical and recovered source CSVs
# may use different valid ISO-8601 timestamp representations.
old_times["kickoff_utc"] = pd.to_datetime(
    old_times["kickoff_utc"],
    utc=True,
    errors="raise",
    format="ISO8601"
)
new_times["kickoff_utc"] = pd.to_datetime(
    new_times["kickoff_utc"],
    utc=True,
    errors="raise",
    format="ISO8601"
)

times = pd.concat(
    [old_times, new_times],
    ignore_index=True
)

print("\n=== EXACT STEP 73 KICKOFF CHECKS ===")
print("Total map rows:", len(times))
print("Expected map rows:", 21878)
print("Duplicate identities:", int(times.duplicated(KEYS).sum()))
print("Null kickoff values:", int(times.kickoff_utc.isna().sum()))
print("Historical rows:", int(times.season.lt(2026).sum()))
print("2026 rows:", int(times.season.eq(2026).sum()))

require(
    len(times) == 21878,
    "Combined kickoff map count mismatch"
)
require(
    not times.duplicated(KEYS).any()
    and times.kickoff_utc.notna().all(),
    "Combined kickoff map integrity failure"
)
require(
    not times.duplicated(
        ["team", "kickoff_utc"]
    ).any(),
    "Duplicate same-team kickoff"
)

raw = raw.merge(
    times,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

require(
    raw.kickoff_utc.notna().all(),
    "Source kickoff coverage incomplete"
)
require(
    raw.kickoff_utc.lt(
        pd.Timestamp.now(tz="UTC")
    ).all(),
    "Source includes future kickoffs"
)

print("Kickoff matched:", raw.kickoff_utc.notna().sum())
print("\nBuilding all 267 features...")

rebuilt, built_columns = build_features(raw)
rebuilt = normalize(rebuilt)

require(
    len(rebuilt) == 21878
    and not rebuilt.duplicated(KEYS).any(),
    "Rebuilt feature coverage failure"
)
require(
    len(built_columns) == 267
    and set(built_columns) == set(features),
    "Rebuilt feature contract mismatch"
)

# Verify that all original game/team identities survive.
original = normalize(pd.read_parquet(ORIGINAL))
identity = rebuilt[KEYS].merge(
    original[KEYS],
    on=KEYS,
    how="outer",
    indicator=True,
    validate="one_to_one"
)
require(
    len(identity) == 21878
    and identity["_merge"].eq("both").all(),
    "Rebuild does not match original identities"
)

print("Identity coverage: PASS")

# Independently verify point history, including opponent defense.
history = rebuilt.sort_values(
    ["team", "kickoff_utc", "game_id"],
    kind="mergesort"
).copy()

group = history.groupby("team", sort=False)

expected = pd.DataFrame(index=history.index)
expected["off_points_lag1"] = group["points"].shift(1)

def prior_average(series, window=None):
    previous = series.shift(1)
    if window is None:
        return previous.expanding(min_periods=1).mean()
    return previous.rolling(
        window, min_periods=1
    ).mean()

for suffix, window in [
    ("avg3", 3), ("avg5", 5),
    ("career_avg", None)
]:
    expected[f"off_points_{suffix}"] = (
        group["points"].transform(
            lambda s: prior_average(s, window)
        )
    )

expected["off_points_season_avg"] = (
    history.groupby(
        ["team", "season"], sort=False
    )["points"].transform(
        lambda s: prior_average(s)
    )
)

defense = history[
    ["game_id", "team", "season", "opponent_points"]
].copy()
def_group = defense.groupby("team", sort=False)

defense["def_allowed_points_lag1"] = (
    def_group["opponent_points"].shift(1)
)

for suffix, window in [
    ("avg3", 3), ("avg5", 5),
    ("career_avg", None)
]:
    defense[f"def_allowed_points_{suffix}"] = (
        def_group["opponent_points"].transform(
            lambda s: prior_average(s, window)
        )
    )

defense["def_allowed_points_season_avg"] = (
    defense.groupby(
        ["team", "season"], sort=False
    )["opponent_points"].transform(
        lambda s: prior_average(s)
    )
)

def_cols = [
    "def_allowed_points_lag1",
    "def_allowed_points_avg3",
    "def_allowed_points_avg5",
    "def_allowed_points_career_avg",
    "def_allowed_points_season_avg",
]

defense = defense.rename(columns={
    "team": "opponent"
})

aligned = history.merge(
    defense[["game_id", "opponent"] + def_cols],
    on=["game_id", "opponent"],
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=("", "_expected"),
)

require(
    aligned["_merge"].eq("both").all(),
    "Opponent defense history join incomplete"
)

# Compare all ten independently computed point-history features.
expected_off = expected.loc[history.index]
mismatches = {}

for col in [
    "off_points_lag1",
    "off_points_avg3",
    "off_points_avg5",
    "off_points_career_avg",
    "off_points_season_avg",
]:
    mismatches[col] = int(
        (~same(history[col], expected_off[col])).sum()
    )

for col in def_cols:
    mismatches[col] = int(
        (~same(
            aligned[col],
            aligned[f"{col}_expected"]
        )).sum()
    )

print("\n=== INDEPENDENT POINT-HISTORY AUDIT ===")
for name, count in mismatches.items():
    print(f"{name}: {count} mismatches")

require(
    all(count == 0 for count in mismatches.values()),
    "Chronology feature audit failed"
)

# Compare all 267 feature columns with the original dataset.
paired = rebuilt[
    KEYS + features
].merge(
    original[KEYS + features],
    on=KEYS,
    how="inner",
    suffixes=("_new", "_old"),
    validate="one_to_one"
)

any_change = np.zeros(len(paired), dtype=bool)
records = []

for feature in features:
    changed = ~same(
        paired[f"{feature}_new"],
        paired[f"{feature}_old"]
    )
    any_change |= changed

    records.append({
        "feature": feature,
        "changed_rows": int(changed.sum()),
        "changed_pct": round(
            float(changed.mean() * 100), 6
        )
    })

differences = pd.DataFrame(records).sort_values(
    ["changed_rows", "feature"],
    ascending=[False, True]
)

paired["any_changed"] = any_change

season_summary = paired.groupby("season").agg(
    rows=("game_id", "size"),
    changed_rows=("any_changed", "sum")
).reset_index()
season_summary["changed_pct"] = (
    season_summary.changed_rows /
    season_summary.rows * 100
)

# All validation gates passed. Write isolated outputs.
OUT.mkdir(parents=True, exist_ok=False)

rebuilt_file = OUT / "features_2014_2026_corrected.parquet"
differences_file = OUT / "feature_differences.csv"
season_file = OUT / "season_change_summary.csv"
manifest_file = OUT / "manifest.json"

rebuilt.to_parquet(rebuilt_file, index=False)
differences.to_csv(differences_file, index=False)
season_summary.to_csv(season_file, index=False)

manifest = {
    "schema": "cfb_full_chronology_rebuild_v2",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "seasons": [2014, 2026],
    "team_game_rows": len(rebuilt),
    "games": int(rebuilt.game_id.nunique()),
    "feature_count": len(features),
    "changed_rows": int(any_change.sum()),
    "changed_features": int(
        differences.changed_rows.gt(0).sum()
    ),
    "point_history_mismatches": mismatches,
    "source_sha256": {
        str(p.relative_to(ROOT)): sha(p)
        for p in inputs + historical_paths
    },
    "outputs_sha256": {
        p.name: sha(p)
        for p in (
            rebuilt_file,
            differences_file,
            season_file
        )
    },
    "week8_9_second_kickoff_source_confirmed": False,
    "historical_source_availability_certified": False,
    "production_approved": False,
    "research_only": True
}

with manifest_file.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print("\n=== STEP 73 RESULTS ===")
print("Rebuilt rows:", len(rebuilt))
print("Rebuilt games:", rebuilt.game_id.nunique())
print("Features:", len(features))
print("Rows with changed features:", int(any_change.sum()))
print(
    "Features with differences:",
    int(differences.changed_rows.gt(0).sum())
)

print("\n=== CHANGES BY SEASON ===")
print(season_summary.to_string(index=False))

print("\n=== TOP TEN CHANGED FEATURES ===")
print(differences.head(10).to_string(index=False))

print("\nSaved:", OUT.relative_to(ROOT))
print("Production files modified: NO")
print("Ridge v4.5 modified: NO")
print("Historical PIT certified: NO")
