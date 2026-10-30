"""Step 53: Isolated kickoff-ordered feature reconstruction."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from cfb_quant.features.matchup_engine import build_features

ROOT = Path.cwd()

ORIGINAL = ROOT / (
    "data/processed/cfb_matchup_features_20261029.parquet"
)
FEATURE_MANIFEST = ROOT / (
    "reports/cfb_matchup_feature_manifest_20261029.json"
)
KICKOFF = ROOT / (
    "reports/spread_total_schedule_chronology_v2/"
    "team_game_kickoff_map.csv"
)

OUT = ROOT / "reports/research/chronology_rebuild_v1"
REBUILT = OUT / "features_kickoff_corrected.parquet"
DIFFERENCES = OUT / "feature_differences.csv"
EXCLUDED = OUT / "original_only_rows.csv"
SUMMARY = OUT / "manifest.json"

KEYS = ["game_id", "season", "team", "home_away"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(frame):
    out = frame.copy()
    out["game_id"] = (
        out.game_id.astype("string")
        .str.replace(r"\.0$", "", regex=True)
    )
    out["season"] = pd.to_numeric(
        out.season, errors="raise"
    ).astype(int)
    out["team"] = out.team.astype(str).str.strip()
    out["home_away"] = (
        out.home_away.astype(str).str.strip().str.lower()
    )
    return out


print("\n=== STEP 53: CHRONOLOGY REBUILD ===")

require(
    all(p.is_file() for p in [
        ORIGINAL, FEATURE_MANIFEST, KICKOFF
    ]),
    "Required original artifact or kickoff map missing"
)

require(
    not OUT.exists(),
    "Research output already exists; refusing overwrite"
)

original_manifest = json.loads(
    FEATURE_MANIFEST.read_text(encoding="utf-8-sig")
)

features = original_manifest["feature_columns"]

require(
    len(features) == 267 and len(set(features)) == 267,
    "Original feature contract changed"
)

sources = sorted(
    ROOT.glob(original_manifest["input_glob"])
)

require(
    len(sources) == 13,
    "Unexpected number of source-year files"
)

raw = pd.concat(
    [pd.read_parquet(p) for p in sources],
    ignore_index=True
)

raw = normalize(raw)

schedule = normalize(pd.read_csv(KICKOFF))

require(
    set(KEYS + ["kickoff_utc"]).issubset(schedule),
    "Invalid kickoff map schema"
)

require(
    raw[KEYS].notna().all().all(),
    "Missing raw team-game identity"
)

require(
    not raw.duplicated(KEYS).any(),
    "Duplicate raw identities"
)

require(
    not schedule.duplicated(KEYS).any(),
    "Duplicate kickoff map identities"
)

raw = raw.drop(
    columns=["kickoff_utc"], errors="ignore"
).merge(
    schedule[KEYS + ["kickoff_utc"]],
    on=KEYS,
    how="left",
    validate="one_to_one"
)

raw["kickoff_utc"] = pd.to_datetime(
    raw["kickoff_utc"],
    utc=True,
    errors="coerce"
)

require(
    raw.kickoff_utc.notna().all(),
    "Not all raw games have verified kickoff timestamps"
)

require(
    raw["points"].notna().all()
    and raw["opponent_points"].notna().all(),
    "Raw source includes games without complete scores"
)

require(
    (
        raw.kickoff_utc
        < pd.Timestamp.now(tz="UTC")
    ).all(),
    "Raw source includes games that have not kicked off"
)

print("Raw rows:", len(raw))
print("Source years:", len(sources))
print("Kickoff coverage: 100%")

print("\nBuilding corrected features...")

rebuilt, built_columns = build_features(raw)

require(
    len(rebuilt) == len(raw),
    "Feature builder lost or added rows"
)

require(
    set(built_columns) == set(features)
    and len(built_columns) == len(features),
    "Rebuilt feature names differ from original contract"
)

rebuilt = normalize(rebuilt)

require(
    not rebuilt.duplicated(KEYS).any(),
    "Duplicate rebuilt identities"
)

require(
    "kickoff_utc" in rebuilt
    and rebuilt.kickoff_utc.notna().all(),
    "Rebuild lost kickoff evidence"
)

original = normalize(pd.read_parquet(ORIGINAL))

require(
    not original.duplicated(KEYS).any(),
    "Duplicate original identities"
)

paired = rebuilt[
    KEYS + features
].merge(
    original[KEYS + features],
    on=KEYS,
    how="inner",
    suffixes=("_corrected", "_original"),
    validate="one_to_one"
)

require(
    len(paired) == len(rebuilt),
    "Not all reconstructed rows match original identities"
)

excluded = original[KEYS].merge(
    rebuilt[KEYS],
    on=KEYS,
    how="left",
    indicator=True,
    validate="one_to_one"
)

excluded = excluded.loc[
    excluded["_merge"].eq("left_only"),
    KEYS
].copy()

require(
    len(original) == len(rebuilt) + len(excluded),
    "Original/rebuilt coverage accounting mismatch"
)

print("\nComparing all 267 features...")

difference_records = []
any_changed = np.zeros(len(paired), dtype=bool)

for feature in features:
    left = pd.to_numeric(
        paired[f"{feature}_corrected"], errors="coerce"
    ).to_numpy(dtype=float)

    right = pd.to_numeric(
        paired[f"{feature}_original"], errors="coerce"
    ).to_numpy(dtype=float)

    equal = np.isclose(
        left, right,
        rtol=1e-9,
        atol=1e-8,
        equal_nan=True,
    )

    changed = ~equal
    any_changed |= changed

    difference_records.append({
        "feature": feature,
        "changed_rows": int(changed.sum()),
        "changed_pct": float(changed.mean() * 100),
        "new_missing": int(np.isnan(left).sum()),
        "old_missing": int(np.isnan(right).sum()),
    })

differences = pd.DataFrame(difference_records)

# Output is written only after all core validations pass.
OUT.mkdir(parents=True, exist_ok=False)

rebuilt.to_parquet(REBUILT, index=False)

differences.sort_values(
    ["changed_rows", "feature"],
    ascending=[False, True]
).to_csv(DIFFERENCES, index=False)

excluded.to_csv(EXCLUDED, index=False)

summary = {
    "schema": "cfb_chronology_rebuild_v1",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "original_rows": len(original),
    "raw_source_rows": len(raw),
    "rebuilt_rows": len(rebuilt),
    "original_only_rows": len(excluded),
    "paired_comparison_rows": len(paired),
    "feature_count": len(features),
    "features_with_differences": int(
        differences.changed_rows.gt(0).sum()
    ),
    "rows_with_any_feature_difference": int(
        any_changed.sum()
    ),
    "source_file_sha256": {
        str(p.relative_to(ROOT)): sha(p)
        for p in sources
    },
    "kickoff_map_sha256": sha(KICKOFF),
    "original_dataset_sha256": sha(ORIGINAL),
    "original_manifest_sha256": sha(FEATURE_MANIFEST),
    "rebuilt_dataset_sha256": sha(REBUILT),
    "differences_sha256": sha(DIFFERENCES),
    "excluded_rows_sha256": sha(EXCLUDED),
    "chronology_policy": "verified_kickoff_utc",
    "historical_source_availability_certified": False,
    "2026_source_availability_certified": False,
    "production_approved": False,
    "research_only": True,
}

with SUMMARY.open("x", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

print("\n=== REBUILD RESULTS ===")
print("Original rows:", len(original))
print("Rebuilt rows:", len(rebuilt))
print("Original-only rows:", len(excluded))
print("Features compared:", len(features))
print(
    "Features with differences:",
    summary["features_with_differences"]
)
print(
    "Rows with any feature difference:",
    summary["rows_with_any_feature_difference"]
)

print("\n=== TOP TEN CHANGED FEATURES ===")
print(
    differences.sort_values(
        "changed_rows", ascending=False
    ).head(10).to_string(index=False)
)

print("\nOutput:", OUT.relative_to(ROOT))
print("Production modified: NO")
print("Original dataset modified: NO")
print("Source availability certified: NO")
