"""Step 72: Complete isolated 2026 raw-source recovery."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from cfb_quant.data.build_team_games import parse_file

ROOT = Path.cwd()

WEEK7 = ROOT / "reports/research/week7_source_recovery_v1"
CAPTURE = ROOT / "reports/research/cfbd_week8_9_capture_v1"

ORIGINAL = ROOT / (
    "data/processed/cfb_matchup_features_20261029.parquet"
)

SCHEDULE_A = ROOT / (
    "reports/missing_games_evidence_20261029/"
    "raw_schedule_result_evidence.csv"
)

SCHEDULE_B = ROOT / (
    "reports/v58_development/chronology_repair_v2/"
    "team_game_kickoff_map_v58.csv"
)

OUT = ROOT / "reports/research/full_2026_source_recovery_v1"

KEYS = ["game_id", "season", "team", "home_away"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(frame):
    df = frame.copy()
    df["game_id"] = (
        df["game_id"].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    df["season"] = pd.to_numeric(
        df["season"], errors="raise"
    ).astype(int)
    df["team"] = df["team"].astype(str).str.strip()
    df["home_away"] = (
        df["home_away"].astype(str)
        .str.strip().str.lower()
    )
    return df


def schedule_table(path, field):
    raw = pd.read_csv(path)

    require(
        {"game_id", field}.issubset(raw.columns),
        f"Missing schedule columns: {path}"
    )

    raw["game_id"] = (
        raw["game_id"].astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    raw["kickoff_utc"] = pd.to_datetime(
        raw[field], utc=True, errors="coerce"
    )

    return raw[["game_id", "kickoff_utc"]]


def unique_times(frame, target_ids, label):
    matched = frame.loc[
        frame.game_id.isin(target_ids)
    ].copy()

    require(
        matched.kickoff_utc.notna().all(),
        f"{label}: null target kickoff"
    )

    conflicts = (
        matched.groupby("game_id")["kickoff_utc"]
        .nunique().gt(1)
    )

    require(
        not conflicts.any(),
        f"{label}: conflicting target kickoff"
    )

    result = matched.drop_duplicates()

    require(
        set(result.game_id) == target_ids,
        f"{label}: incomplete target coverage"
    )

    return result


print("\n=== STEP 72: COMPLETE 2026 SOURCE RECOVERY ===")

base_file = WEEK7 / "team_games_2026_research.parquet"
base_times = WEEK7 / "team_game_kickoff_map.csv"
base_manifest_file = WEEK7 / "manifest.json"
capture_manifest_file = CAPTURE / "manifest.json"

inputs = [
    base_file, base_times, base_manifest_file,
    capture_manifest_file, ORIGINAL, SCHEDULE_A
]

require(
    all(path.is_file() for path in inputs),
    "Required recovery input missing"
)

require(
    not OUT.exists(),
    "Output already exists; refusing overwrite"
)

base_manifest = json.loads(
    base_manifest_file.read_text(encoding="utf-8-sig")
)

capture_manifest = json.loads(
    capture_manifest_file.read_text(encoding="utf-8-sig")
)

for filename, digest in base_manifest["outputs_sha256"].items():
    require(
        sha(WEEK7 / filename) == digest,
        f"Week 7 artifact hash mismatch: {filename}"
    )

require(
    sha(ORIGINAL) == capture_manifest["source_original_sha256"],
    "Original feature source changed"
)

require(
    sha(base_file) == capture_manifest["source_recovered_sha256"],
    "Week 7 source differs from API capture baseline"
)

base = normalize(pd.read_parquet(base_file))
original = normalize(pd.read_parquet(ORIGINAL))

original = original.loc[
    original.season.eq(2026)
].copy()

require(len(base) == 1020, "Expected 1,020 base rows")
require(len(original) == 1140, "Expected 1,140 target rows")

require(
    len(capture_manifest["captures"]) == 2
    and {
        int(e["week"]) for e in capture_manifest["captures"]
    } == {8, 9},
    "Unexpected capture manifest weeks"
)

parsed_frames = []
capture_completion = {}

for entry in sorted(
    capture_manifest["captures"],
    key=lambda x: x["week"]
):
    week = int(entry["week"])

    raw_path = CAPTURE / entry["raw_response_file"]
    selected_path = CAPTURE / entry["selected_file"]

    require(
        sha(raw_path) == entry["raw_response_sha256"]
        and sha(selected_path) == entry["selected_sha256"],
        f"Week {week}: captured source hash mismatch"
    )

    require(
        not entry["missing_game_ids"]
        and not entry["invalid_pair_ids"]
        and not entry["score_mismatch_ids"],
        f"Week {week}: capture integrity exceptions"
    )

    frame = normalize(
        pd.DataFrame(parse_file(selected_path))
    )

    require(
        len(frame) == 2 * entry["matched_games"]
        and frame.week.eq(week).all(),
        f"Week {week}: parser coverage mismatch"
    )

    parsed_frames.append(frame)

    capture_completion[week] = pd.to_datetime(
        entry["request_completed_utc"],
        utc=True
    )

new = pd.concat(parsed_frames, ignore_index=True)

require(
    len(new) == 120
    and new.game_id.nunique() == 60,
    "Expected 120 recovered rows from 60 games"
)

require(
    set(new.columns) == set(base.columns),
    "Raw schema mismatch"
)

require(
    not base.duplicated(KEYS).any()
    and not new.duplicated(KEYS).any(),
    "Duplicate raw identities"
)

require(
    set(base.game_id).isdisjoint(set(new.game_id)),
    "Recovered game IDs overlap existing history"
)

# Ensure all 1,140 identities are exactly the intended ones.
combined = pd.concat(
    [base, new[base.columns]],
    ignore_index=True
)

require(
    len(combined) == 1140
    and combined.game_id.nunique() == 570,
    "Combined row or game count mismatch"
)

require(
    not combined.duplicated(KEYS).any(),
    "Combined duplicate identities"
)

coverage = combined[KEYS].merge(
    original[KEYS],
    on=KEYS,
    how="outer",
    indicator=True,
    validate="one_to_one"
)

require(
    len(coverage) == 1140
    and coverage["_merge"].eq("both").all(),
    "Combined identities do not equal original target"
)

# Validate exact team-specific scoring.
scores = combined[
    KEYS + ["points", "opponent_points"]
].merge(
    original[KEYS + ["points", "opponent_points"]],
    on=KEYS,
    suffixes=("_raw", "_original"),
    validate="one_to_one"
)

for column in ("points", "opponent_points"):
    a = scores[f"{column}_raw"].to_numpy(dtype=float)
    b = scores[f"{column}_original"].to_numpy(dtype=float)

    require(
        np.isclose(
            a, b, rtol=0, atol=1e-8,
            equal_nan=False
        ).all(),
        f"Original scoring mismatch: {column}"
    )

# Validate full game pairs, team identities, and scores.
for game_id, game in combined.groupby("game_id"):
    home = game.loc[game.home_away.eq("home")]
    away = game.loc[game.home_away.eq("away")]

    require(
        len(home) == 1 and len(away) == 1,
        f"Game {game_id}: incomplete home/away pair"
    )

    h, a = home.iloc[0], away.iloc[0]

    require(
        h.team == a.opponent
        and a.team == h.opponent,
        f"Game {game_id}: opponent mismatch"
    )

    require(
        np.isclose(h.points, a.opponent_points)
        and np.isclose(a.points, h.opponent_points),
        f"Game {game_id}: reciprocal score mismatch"
    )

print("Raw game identities and scores: PASS")

# Preserve the existing verified kickoff map.
times_old = normalize(pd.read_csv(base_times))

times_old["kickoff_utc"] = pd.to_datetime(
    times_old["kickoff_utc"],
    utc=True,
    errors="coerce"
)

require(
    len(times_old) == 1020
    and not times_old.duplicated(KEYS).any()
    and times_old.kickoff_utc.notna().all(),
    "Week 7 kickoff map invalid"
)

# Obtain missing Weeks 8-9 kickoff evidence.
target_ids = set(new.game_id)

times_a = unique_times(
    schedule_table(SCHEDULE_A, "start_date"),
    target_ids,
    "schedule A"
)

# Optional independent source must agree where available.
if SCHEDULE_B.is_file():
    other = schedule_table(SCHEDULE_B, "kickoff_utc")
    other = other.loc[other.game_id.isin(target_ids)]

    require(
        not other.groupby("game_id")["kickoff_utc"]
        .nunique().gt(1).any(),
        "Schedule B contains conflicting timestamps"
    )

    other = other.drop_duplicates().rename(
        columns={"kickoff_utc": "other_kickoff"}
    )

    cross = times_a.merge(
        other,
        on="game_id",
        how="left",
        validate="one_to_one"
    )

    disagree = (
        cross.other_kickoff.notna()
        & cross.kickoff_utc.ne(cross.other_kickoff)
    )

    require(
        not disagree.any(),
        "Independent kickoff source disagreement"
    )

    print(
        "Independent kickoff cross-checks:",
        int(cross.other_kickoff.notna().sum())
    )

times_new = new[KEYS].merge(
    times_a,
    on="game_id",
    how="left",
    validate="many_to_one"
)

require(
    times_new.kickoff_utc.notna().all(),
    "Missing recovered kickoff timestamps"
)

# A successful historical capture cannot predate game kickoff.
for week in (8, 9):
    ids_for_week = set(
        new.loc[new.week.eq(week), "game_id"]
    )

    relevant = times_a.loc[
        times_a.game_id.isin(ids_for_week)
    ]

    require(
        relevant.kickoff_utc.lt(
            capture_completion[week]
        ).all(),
        f"Week {week}: kickoff at/after capture"
    )

times = pd.concat(
    [times_old[KEYS + ["kickoff_utc"]],
     times_new[KEYS + ["kickoff_utc"]]],
    ignore_index=True
)

require(
    len(times) == 1140
    and not times.duplicated(KEYS).any()
    and not times.duplicated(
        ["team", "kickoff_utc"]
    ).any()
    and times.kickoff_utc.notna().all(),
    "Final kickoff-map integrity failure"
)

print("Kickoff coverage:", len(times), "/ 1140")
print("Kickoff chronology: PASS")

# All validations succeeded. Write isolated artifacts.
OUT.mkdir(parents=True, exist_ok=False)

raw_out = OUT / "team_games_2026_complete.parquet"
times_out = OUT / "team_game_kickoff_map.csv"
manifest_out = OUT / "manifest.json"

combined.to_parquet(raw_out, index=False)

times.sort_values(
    ["kickoff_utc", "game_id", "team"]
).to_csv(times_out, index=False)

manifest = {
    "schema": "cfb_full_2026_source_recovery_v1",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "season": 2026,
    "team_game_rows": len(combined),
    "games": combined.game_id.nunique(),
    "existing_and_week7_rows": len(base),
    "week8_9_recovered_rows": len(new),
    "source_sha256": {
        str(p.relative_to(ROOT)): sha(p)
        for p in inputs
    },
    "capture_manifest_sha256": sha(capture_manifest_file),
    "outputs_sha256": {
        raw_out.name: sha(raw_out),
        times_out.name: sha(times_out),
    },
    "historical_source_availability_certified": False,
    "production_approved": False,
    "research_only": True,
}

if SCHEDULE_B.is_file():
    manifest["source_sha256"][
        str(SCHEDULE_B.relative_to(ROOT))
    ] = sha(SCHEDULE_B)

with manifest_out.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print("\n=== STEP 72 RESULTS ===")
print("Complete 2026 rows:", len(combined))
print("Complete 2026 games:", combined.game_id.nunique())
print("Kickoff rows:", len(times))
print("Remaining missing identities: 0")
print("Saved:", OUT.relative_to(ROOT))
print("Production source modified: NO")
print("Ridge v4.5 modified: NO")
print("Historical PIT certified: NO")
