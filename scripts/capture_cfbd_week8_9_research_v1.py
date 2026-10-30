"""Step 70: Isolated CFBD Week 8-9 team-stat acquisition."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

from cfb_quant.data.download_history import cfbd_get

ROOT = Path.cwd()

ORIGINAL = ROOT / (
    "data/processed/cfb_matchup_features_20261029.parquet"
)

RECOVERED = ROOT / (
    "reports/research/week7_source_recovery_v1/"
    "team_games_2026_research.parquet"
)

OUT = ROOT / "reports/research/cfbd_week8_9_capture_v1"

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def game_ids(series):
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")

print("\n=== STEP 70: CFBD RAW RECOVERY ===")

require(ORIGINAL.is_file(), "Original feature file missing")
require(RECOVERED.is_file(), "Week 7 recovery missing")
require(not OUT.exists(), "Output already exists; refusing overwrite")

original = pd.read_parquet(
    ORIGINAL,
    columns=[
        "game_id", "season", "week",
        "team", "points"
    ],
)

original = original.loc[
    original["season"].eq(2026)
].copy()

original["game_id"] = game_ids(original["game_id"])

recovered = pd.read_parquet(
    RECOVERED, columns=["game_id"]
)

existing = set(game_ids(recovered["game_id"]))

targets = original.loc[
    ~original["game_id"].isin(existing)
].copy()

target_by_week = {
    week: set(
        targets.loc[
            targets["week"].eq(week), "game_id"
        ]
    )
    for week in (8, 9)
}

require(
    len(target_by_week[8]) == 56,
    "Expected 56 missing Week 8 games"
)
require(
    len(target_by_week[9]) == 4,
    "Expected 4 missing Week 9 games"
)
require(
    targets["game_id"].nunique() == 60,
    "Unexpected target coverage"
)

print("Target Week 8 games:", len(target_by_week[8]))
print("Target Week 9 games:", len(target_by_week[9]))

# Only create a new research output directory.
OUT.mkdir(parents=True, exist_ok=False)

manifest = {
    "schema": "cfbd_week8_9_capture_v1",
    "created_at_utc": utc_now(),
    "endpoint": "/games/teams",
    "season": 2026,
    "source_original_sha256": sha(ORIGINAL),
    "source_recovered_sha256": sha(RECOVERED),
    "captures": [],
    "research_only": True,
    "historical_source_availability_certified": False,
    "production_approved": False,
}

for week in (8, 9):
    print(f"\n=== FETCHING 2026 WEEK {week} ===")

    params = {
        "year": 2026,
        "week": week,
        "seasonType": "regular",
    }

    started = utc_now()

    # Existing repository client handles authentication
    # and HTTP failures. Do not print credentials.
    data = cfbd_get(
        "/games/teams",
        params=params,
        attempts=3,
    )

    completed = utc_now()

    require(
        isinstance(data, list),
        f"Week {week}: expected a list response"
    )

    raw_path = OUT / (
        f"cfbd_team_stats_week_{week:02d}_raw.json"
    )

    # Preserve the complete response before evaluating it.
    write_json(raw_path, data)

    by_id = {}

    for game in data:
        require(
            isinstance(game, dict),
            f"Week {week}: malformed game record"
        )

        gid = str(game.get("id", "")).removesuffix(".0")

        require(
            gid and gid not in by_id,
            f"Week {week}: missing or duplicate game ID"
        )

        by_id[gid] = game

    target_ids = target_by_week[week]

    found = target_ids & set(by_id)
    missing = target_ids - set(by_id)

    selected = [
        by_id[gid] for gid in sorted(found)
    ]

    # Use standard filenames understood by parse_file().
    selected_path = OUT / (
        f"team_stats_2026_regular_week_{week:02d}.json"
    )

    write_json(selected_path, selected)

    # Independently compare returned team-score records
    # to the reference game scores.
    score_errors = []
    invalid_pairs = []

    for gid in sorted(found):
        game = by_id[gid]
        teams = game.get("teams")

        if (
            not isinstance(teams, list)
            or len(teams) != 2
            or set(
                str(t.get("homeAway", "")).lower()
                for t in teams
                if isinstance(t, dict)
            ) != {"home", "away"}
        ):
            invalid_pairs.append(gid)
            continue

        expected = targets.loc[
            targets["game_id"].eq(gid)
        ]

        expected_scores = sorted(
            int(x) for x in expected["points"]
        )

        try:
            returned_scores = sorted(
                int(float(t["points"]))
                for t in teams
            )
        except (TypeError, ValueError, KeyError):
            score_errors.append(gid)
            continue

        if returned_scores != expected_scores:
            score_errors.append(gid)

    entry = {
        "week": week,
        "request_started_utc": started,
        "request_completed_utc": completed,
        "params": params,
        "raw_response_file": raw_path.name,
        "raw_response_sha256": sha(raw_path),
        "selected_file": selected_path.name,
        "selected_sha256": sha(selected_path),
        "api_returned_games": len(data),
        "target_games": len(target_ids),
        "matched_games": len(found),
        "missing_game_ids": sorted(missing),
        "invalid_pair_ids": invalid_pairs,
        "score_mismatch_ids": score_errors,
    }

    manifest["captures"].append(entry)

    # Save a checkpoint after each completed request.
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )

    print("API returned games:", len(data))
    print("Target games matched:", len(found))
    print("Target games missing:", len(missing))
    print("Invalid home/away pairs:", len(invalid_pairs))
    print("Score mismatches:", len(score_errors))
    print("Raw response saved:", raw_path.name)

print("\n=== STEP 70 FINAL COVERAGE ===")

matched_total = sum(
    e["matched_games"] for e in manifest["captures"]
)

missing_total = sum(
    len(e["missing_game_ids"])
    for e in manifest["captures"]
)

invalid_total = sum(
    len(e["invalid_pair_ids"])
    + len(e["score_mismatch_ids"])
    for e in manifest["captures"]
)

print("Target games:", 60)
print("Recovered game identities:", matched_total)
print("Missing game identities:", missing_total)
print("Pair/score exceptions:", invalid_total)
print("Output:", OUT.relative_to(ROOT))

if (
    matched_total == 60
    and missing_total == 0
    and invalid_total == 0
):
    print("PASS: complete game identity and score coverage.")
    print("NEXT: validate full statistical parser output.")
else:
    print("PARTIAL OR INCONSISTENT CAPTURE.")
    print("Review the manifest before any dataset merge.")

print("\nSource publication timing certified: NO")
print("Production raw files modified: NO")
print("Ridge champion modified: NO")
