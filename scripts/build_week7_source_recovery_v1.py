"""Step 65: Isolated, validated Week 7 source recovery."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path.cwd()

CURRENT = ROOT / "data/processed/team_games_2026.parquet"

CANDIDATE = ROOT / (
    "reports/v58_development/team_history_refresh_v2/"
    "team_games_2026_candidate.parquet"
)

SOURCE_MANIFEST = CANDIDATE.parent / "manifest.json"

ORIGINAL = ROOT / (
    "data/processed/cfb_matchup_features_20261029.parquet"
)

OLD_MAP = ROOT / (
    "reports/spread_total_schedule_chronology_v2/"
    "team_game_kickoff_map.csv"
)

SCHEDULE_A = ROOT / (
    "reports/missing_games_evidence_20261029/"
    "raw_schedule_result_evidence.csv"
)

SCHEDULE_B = ROOT / (
    "reports/v58_development/chronology_repair_v2/"
    "team_game_kickoff_map_v58.csv"
)

OUT = ROOT / "reports/research/week7_source_recovery_v1"

KEYS = ["game_id", "season", "team", "home_away"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(df):
    result = df.copy()
    result["game_id"] = (
        result["game_id"].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    result["season"] = pd.to_numeric(
        result["season"], errors="raise"
    ).astype(int)
    result["team"] = result["team"].astype(str).str.strip()
    result["home_away"] = (
        result["home_away"].astype(str)
        .str.strip().str.lower()
    )
    return result


def time_table(path, field):
    df = pd.read_csv(path)

    require(
        {"game_id", field}.issubset(df.columns),
        f"Invalid schedule schema: {path}"
    )

    df["game_id"] = (
        df["game_id"].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    df["kickoff_utc"] = pd.to_datetime(
        df[field], utc=True, errors="coerce"
    )

    require(
        not df.loc[
            df["game_id"].notna(), "kickoff_utc"
        ].isna().any(),
        f"Invalid timestamps: {path}"
    )

    require(
        not df.groupby("game_id")["kickoff_utc"]
        .nunique().gt(1).any(),
        f"Conflicting timestamps: {path}"
    )

    return df[
        ["game_id", "kickoff_utc"]
    ].drop_duplicates()


print("\n=== STEP 65: WEEK 7 SOURCE RECOVERY ===")

inputs = [
    CURRENT, CANDIDATE, SOURCE_MANIFEST,
    ORIGINAL, OLD_MAP, SCHEDULE_A, SCHEDULE_B
]

require(
    all(p.is_file() for p in inputs),
    "Required source file missing"
)

require(
    not OUT.exists(),
    "Output directory already exists; refusing overwrite"
)

current = normalize(pd.read_parquet(CURRENT))
candidate = normalize(pd.read_parquet(CANDIDATE))
original = normalize(pd.read_parquet(ORIGINAL))
old_map = normalize(pd.read_csv(OLD_MAP))

original = original.loc[
    original["season"].eq(2026)
].copy()

source_manifest = json.loads(
    SOURCE_MANIFEST.read_text(encoding="utf-8-sig")
)

require(
    sha(CANDIDATE) == source_manifest["candidate_sha256"],
    "Candidate source hash mismatch"
)

require(
    set(current.columns) == set(candidate.columns),
    "Raw source schema mismatch"
)

for name, df in [
    ("current", current),
    ("candidate", candidate),
    ("original", original),
    ("old_map", old_map)
]:
    require(
        not df.duplicated(KEYS).any(),
        f"Duplicate identities: {name}"
    )

require(
    len(current) == 910 and len(candidate) == 1020,
    "Unexpected source row counts"
)

# Confirm that the candidate preserves all 910 original rows.
joined = current.merge(
    candidate,
    on=KEYS,
    how="inner",
    suffixes=("_old", "_new"),
    validate="one_to_one"
)

require(
    len(joined) == 910,
    "Candidate does not preserve all original identities"
)

for col in current.columns:
    if col in KEYS:
        continue

    left = joined[f"{col}_old"]
    right = joined[f"{col}_new"]

    if pd.api.types.is_numeric_dtype(current[col]):
        equal = np.isclose(
            pd.to_numeric(left).to_numpy(dtype=float),
            pd.to_numeric(right).to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-8,
            equal_nan=True
        )
    else:
        equal = (
            left.astype("string")
            .eq(right.astype("string"))
            .fillna(False)
            | (left.isna() & right.isna())
        ).to_numpy()

    require(
        bool(np.all(equal)),
        f"Source value changed: {col}"
    )

# Identify the 110 genuinely additional Week 7 rows.
new = candidate.merge(
    current[KEYS],
    on=KEYS,
    how="left",
    indicator=True,
    validate="one_to_one"
)

new = new.loc[
    new["_merge"].eq("left_only")
].drop(columns="_merge")

require(
    len(new) == 110 and new["game_id"].nunique() == 55,
    "Expected 110 rows from 55 additional games"
)

require(
    new["week"].eq(7).all(),
    "Additional games are not all Week 7"
)

require(
    not new[current.columns].isna().any().any(),
    "Recovered source records contain nulls"
)

# Validate scores against the October 29 artifact.
scores = new[
    KEYS + ["points", "opponent_points"]
].merge(
    original[KEYS + ["points", "opponent_points"]],
    on=KEYS,
    suffixes=("_candidate", "_original"),
    validate="one_to_one"
)

require(
    len(scores) == 110,
    "Recovered games not all present in original artifact"
)

for col in ["points", "opponent_points"]:
    require(
        bool(np.isclose(
            scores[f"{col}_candidate"].to_numpy(dtype=float),
            scores[f"{col}_original"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-8
        ).all()),
        f"Score disagreement: {col}"
    )

# Check reciprocal game records.
for game_id, game in candidate.groupby("game_id"):
    home = game.loc[game["home_away"].eq("home")]
    away = game.loc[game["home_away"].eq("away")]

    require(
        len(home) == 1 and len(away) == 1,
        f"Invalid home-away pair: {game_id}"
    )

    h, a = home.iloc[0], away.iloc[0]

    require(
        h["team"] == a["opponent"]
        and a["team"] == h["opponent"],
        f"Opponent mismatch: {game_id}"
    )

    require(
        np.isclose(h["points"], a["opponent_points"])
        and np.isclose(a["points"], h["opponent_points"]),
        f"Reciprocal score mismatch: {game_id}"
    )

# Cross-check recovered kickoff times against both schedules.
schedule_a = time_table(SCHEDULE_A, "start_date")
schedule_b = time_table(SCHEDULE_B, "kickoff_utc")

new_ids = new[["game_id"]].drop_duplicates()

extra_times = (
    new_ids.merge(
        schedule_a.rename(
            columns={"kickoff_utc": "time_a"}
        ),
        on="game_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        schedule_b.rename(
            columns={"kickoff_utc": "time_b"}
        ),
        on="game_id",
        how="left",
        validate="one_to_one"
    )
)

require(
    extra_times[["time_a", "time_b"]].notna().all().all(),
    "Recovered kickoff times missing"
)

require(
    extra_times["time_a"].eq(extra_times["time_b"]).all(),
    "Kickoff schedule disagreement"
)

captured = pd.to_datetime(
    source_manifest["created_utc"],
    utc=True
)

require(
    extra_times["time_a"].lt(captured).all(),
    "Kickoff does not precede recorded capture"
)

# Preserve kickoff data for the existing 910 rows.
old_map["kickoff_utc"] = pd.to_datetime(
    old_map["kickoff_utc"],
    utc=True,
    errors="coerce"
)

existing_times = current[KEYS].merge(
    old_map[KEYS + ["kickoff_utc"]],
    on=KEYS,
    how="left",
    validate="one_to_one"
)

require(
    existing_times["kickoff_utc"].notna().all(),
    "Missing existing kickoff timestamps"
)

recovered_times = new[KEYS].merge(
    extra_times[
        ["game_id", "time_a"]
    ].rename(columns={"time_a": "kickoff_utc"}),
    on="game_id",
    how="left",
    validate="many_to_one"
)

combined_times = pd.concat(
    [existing_times, recovered_times],
    ignore_index=True
)

require(
    len(combined_times) == 1020,
    "Combined kickoff map row count mismatch"
)

require(
    not combined_times.duplicated(KEYS).any(),
    "Duplicate combined kickoff identities"
)

require(
    combined_times["kickoff_utc"].notna().all(),
    "Combined kickoff map contains missing values"
)

require(
    not combined_times.duplicated(
        ["team", "kickoff_utc"]
    ).any(),
    "Duplicate team kickoff times"
)

# All core checks passed. Write an isolated research snapshot.
OUT.mkdir(parents=True, exist_ok=False)

raw_out = OUT / "team_games_2026_research.parquet"
kickoff_out = OUT / "team_game_kickoff_map.csv"
manifest_out = OUT / "manifest.json"

combined = candidate[current.columns].copy()

combined.to_parquet(raw_out, index=False)

combined_times.sort_values(
    ["kickoff_utc", "game_id", "team"]
).to_csv(kickoff_out, index=False)

manifest = {
    "schema": "cfb_week7_source_recovery_v1",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "season": 2026,
    "original_rows": 910,
    "recovered_rows": 110,
    "recovered_games": 55,
    "combined_rows": 1020,
    "remaining_original_only_rows": 120,
    "source_capture_utc": captured.isoformat(),
    "sources_sha256": {
        str(p.relative_to(ROOT)): sha(p)
        for p in inputs
    },
    "outputs_sha256": {
        raw_out.name: sha(raw_out),
        kickoff_out.name: sha(kickoff_out)
    },
    "historical_source_availability_certified": False,
    "production_approved": False,
    "research_only": True
}

with manifest_out.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print("\n=== STEP 65 RESULTS ===")
print("Original rows:", len(current))
print("Recovered Week 7 rows:", len(new))
print("Combined research rows:", len(combined))
print("Combined games:", combined["game_id"].nunique())
print("Verified kickoff rows:", len(combined_times))
print("Remaining original-only rows:", 120)
print("Saved to:", OUT.relative_to(ROOT))

print("\nProduction dataset modified: NO")
print("Ridge champion modified: NO")
print("Research-only: YES")
print("Historical PIT certified: NO")
