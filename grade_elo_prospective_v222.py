import csv
import hashlib
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path.cwd()
LOCK = ROOT / (
    "reports/v58_development/elo_prospective_locks_v221/"
    "20261021T204450423907Z"
)
OUT = ROOT / "reports/v58_development/elo_prospective_grades_v222"
PRED = LOCK / "locked_predictions.csv"
MANIFEST = LOCK / "manifest.json"

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().lower()

def parse_dt(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp missing timezone: " + str(value))
    return dt.astimezone(timezone.utc)

def get_key():
    for k in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
                return v.strip().strip('"').strip("'")
    raise RuntimeError("No CFBD API key found")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
expected_hash = manifest["locked_predictions_sha256"].lower()
assert digest(PRED) == expected_hash, "FROZEN PREDICTION HASH MISMATCH"

for name, expected in manifest["input_hashes"].items():
    source = ROOT / name.replace("\\", "/")
    assert source.is_file(), f"Missing locked input: {name}"
    assert digest(source) == expected.lower(), f"Input changed: {name}"

with PRED.open(newline="", encoding="utf-8-sig") as f:
    locked = list(csv.DictReader(f))

assert len(locked) == manifest["games_locked"], "Lock count mismatch"
ids = [int(row["game_id"]) for row in locked]
assert len(set(ids)) == len(ids), "Duplicate game IDs"

url = "https://api.collegefootballdata.com/games?" + urlencode({
    "year": 2026,
    "seasonType": "regular",
})
req = Request(url, headers={
    "Authorization": "Bearer " + get_key(),
    "Accept": "application/json",
})

with urlopen(req, timeout=45) as response:
    games = json.load(response)

assert isinstance(games, list), "Unexpected CFBD response"
by_id = {int(g["id"]): g for g in games}

graded = []
pending = []
excluded = []

for row in locked:
    gid = int(row["game_id"])
    game = by_id.get(gid)

    if game is None:
        excluded.append((gid, "Game ID absent from API response"))
        continue

    if (
        game.get("homeTeam") != row["home_forecast"]
        or game.get("awayTeam") != row["away_forecast"]
    ):
        excluded.append((gid, "Team name mismatch"))
        continue

    kickoff = parse_dt(row["kickoff_utc_forecast"])
    lock_time = parse_dt(row["locked_at_utc"])
    elo_capture = parse_dt(row["elo_capture_dt"])
    source_mtime = parse_dt(row["forecast_file_mtime_utc"])

    if not (
        lock_time < kickoff
        and elo_capture < kickoff
        and source_mtime < kickoff
    ):
        excluded.append((gid, "Invalid pregame chronology"))
        continue

    if game.get("completed") is not True:
        pending.append(gid)
        continue

    if game.get("homePoints") is None or game.get("awayPoints") is None:
        excluded.append((gid, "Completed game missing score"))
        continue

    actual = int(game["homePoints"]) - int(game["awayPoints"])
    original = float(row["original_margin_home"])
    elo = float(row["elo_margin_home"])

    if not all(math.isfinite(x) for x in (original, elo)):
        excluded.append((gid, "Non-finite forecast"))
        continue

    original_ae = abs(original - actual)
    elo_ae = abs(elo - actual)

    graded.append({
        "game_id": gid,
        "away": row["away_forecast"],
        "home": row["home_forecast"],
        "actual_home_margin": actual,
        "original_home_margin": original,
        "elo_home_margin": elo,
        "original_ae": original_ae,
        "elo_ae": elo_ae,
        "elo_minus_original_ae": elo_ae - original_ae,
    })

n = len(graded)
summary = {
    "lock_sha256": expected_hash,
    "graded_at_utc": datetime.now(timezone.utc).isoformat(),
    "total_locked": len(locked),
    "completed_graded": n,
    "pending": len(pending),
    "excluded": len(excluded),
    "original_mae": (
        sum(x["original_ae"] for x in graded) / n if n else None
    ),
    "elo_mae": (
        sum(x["elo_ae"] for x in graded) / n if n else None
    ),
    "elo_minus_original_mae": (
        sum(x["elo_minus_original_ae"] for x in graded) / n
        if n else None
    ),
    "historical_elo_pit_certified": False,
    "source_feature_cutoffs_certified": False,
    "production_promotion_authorized": False,
}

print("=" * 72)
print("STEP 16 - PROSPECTIVE ELO GRADING")
print("=" * 72)
print("Frozen lock verified:", expected_hash)
print("Locked:", len(locked))
print("Graded:", n)
print("Pending:", len(pending))
print("Excluded:", len(excluded))

for x in graded:
    print(
        f"{x['away']} @ {x['home']} | "
        f"Actual margin {x['actual_home_margin']:+d} | "
        f"Original AE {x['original_ae']:.3f} | "
        f"Elo AE {x['elo_ae']:.3f} | "
        f"Delta {x['elo_minus_original_ae']:+.3f}"
    )

print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print("EXCLUSIONS:", excluded)
print("PENDING IDS:", pending)

# Validate destination before writing anything.
OUT.mkdir(parents=True, exist_ok=True)
output = OUT / "prospective_grade.json"
if output.exists():
    raise RuntimeError("Report already exists; refusing overwrite")

with output.open("x", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "games": graded,
        "pending_game_ids": pending,
        "excluded": excluded,
    }, f, indent=2)

print("\nSAVED:", output)
print("ALL LOCKED FORECASTS UNCHANGED")
