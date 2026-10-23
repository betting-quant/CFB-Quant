import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path.cwd()
PRED = ROOT / "reports/live_predictions_v57/2026-10-22_20261022T211001Z/slate_forecasts.csv"
OUT = ROOT / "reports/postgame_grades_v57/2026-10-22"
EXPECTED_HASH = "73ACBF0D6268133B998F835AB61531B92C675D62D1C7F4F4560E4D3D6E33B0AD"

EXPECTED = {
    401869845: ("James Madison", "App State"),
    401871056: ("Liberty", "Kennesaw State"),
    401862807: ("East Carolina", "Memphis"),
}

def load_key():
    for name in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
                return value.strip().strip('"').strip("'")
    raise RuntimeError("CFBD key missing from environment or project .env")

def check_game(game_id, key):
    url = "https://api.collegefootballdata.com/games?" + urlencode({"id": game_id})
    req = Request(url, headers={
        "Authorization": "Bearer " + key,
        "Accept": "application/json",
        "User-Agent": "CFB-Quant-postgame-audit/1.0",
    })
    with urlopen(req, timeout=30) as response:
        results = json.load(response)

    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"Expected exactly one record for {game_id}")

    result = results[0]
    away, home = EXPECTED[game_id]

    if int(result["id"]) != game_id:
        raise RuntimeError(f"Unexpected game ID for {game_id}")
    if (result["awayTeam"], result["homeTeam"]) != (away, home):
        raise RuntimeError(f"Team mismatch for {game_id}: {result}")
    if result.get("completed") is not True:
        raise RuntimeError(f"Game {game_id} is not marked completed")
    if result.get("homePoints") is None or result.get("awayPoints") is None:
        raise RuntimeError(f"Missing final score for {game_id}")

    return {
        "game_id": game_id,
        "away": away,
        "home": home,
        "away_points": int(result["awayPoints"]),
        "home_points": int(result["homePoints"]),
    }

if not PRED.is_file():
    raise RuntimeError("Locked prediction CSV missing")

source_bytes = PRED.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest().upper()

if source_hash != EXPECTED_HASH:
    raise RuntimeError("LOCKED CSV HASH MISMATCH - refusing grading")

with PRED.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    forecasts = list(reader)

if len(forecasts) != 3:
    raise RuntimeError(f"Expected 3 forecasts, got {len(forecasts)}")

print("LOCKED FORECAST SHA256:", source_hash)
print("AVAILABLE CSV COLUMNS:", columns)

required = {"game_id", "home", "away", "fair_margin_home", "fair_total"}
if not required.issubset(columns):
    raise RuntimeError("Required CSV columns missing: " + str(required - set(columns)))

key = load_key()
actuals = {game_id: check_game(game_id, key) for game_id in EXPECTED}

rows = []

for f in forecasts:
    game_id = int(f["game_id"])
    if game_id not in actuals:
        raise RuntimeError(f"Unexpected forecast game: {game_id}")

    actual = actuals[game_id]
    if (f["away"], f["home"]) != (actual["away"], actual["home"]):
        raise RuntimeError(f"Forecast team mismatch: {game_id}")

    predicted_margin = float(f["fair_margin_home"])
    predicted_total = float(f["fair_total"])
    actual_margin = actual["home_points"] - actual["away_points"]
    actual_total = actual["home_points"] + actual["away_points"]

    rows.append({
        **actual,
        "predicted_home_margin": predicted_margin,
        "actual_home_margin": actual_margin,
        "margin_error": predicted_margin - actual_margin,
        "absolute_margin_error": abs(predicted_margin - actual_margin),
        "predicted_total": predicted_total,
        "actual_total": actual_total,
        "total_error": predicted_total - actual_total,
        "absolute_total_error": abs(predicted_total - actual_total),
        "winner_correct": (
            (predicted_margin > 0 and actual_margin > 0)
            or (predicted_margin < 0 and actual_margin < 0)
        ),
    })

summary = {
    "date": "2026-10-22",
    "game_count": len(rows),
    "margin_mae": sum(r["absolute_margin_error"] for r in rows) / len(rows),
    "total_mae": sum(r["absolute_total_error"] for r in rows) / len(rows),
    "winner_accuracy": sum(r["winner_correct"] for r in rows) / len(rows),
    "source_csv_sha256": source_hash,
    "graded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": "CFBD GET /games?id=<game_id>",
    "notes": [
        "Original locked forecasts were not modified.",
        "Three games are insufficient to establish model calibration.",
        "Errors are signed as forecast minus actual.",
        "This report uses final scores, not updated postgame features.",
    ],
}

OUT.mkdir(parents=True, exist_ok=True)
result_path = OUT / "grade_results.json"

# Exclusive creation: never silently overwrite an audit.
with result_path.open("x", encoding="utf-8") as f:
    json.dump({"summary": summary, "games": rows}, f, indent=2)

print("\n=== CFB-QUANT OCTOBER 22 POSTGAME GRADING ===")

for r in rows:
    print(
        f"{r['away']} @ {r['home']}: "
        f"{r['away_points']}-{r['home_points']} | "
        f"Margin AE={r['absolute_margin_error']:.3f} | "
        f"Total AE={r['absolute_total_error']:.3f} | "
        f"Winner correct={r['winner_correct']}"
    )

print(f"\nMARGIN MAE: {summary['margin_mae']:.4f}")
print(f"TOTAL MAE: {summary['total_mae']:.4f}")
print(f"WINNER ACCURACY: {summary['winner_accuracy']:.1%}")
print(f"AUDIT FILE: {result_path}")
print("LOCKED FORECAST: UNCHANGED")
