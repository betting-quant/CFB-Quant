import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

HIST = ROOT / (
    "reports/v58_development/elo_research_v212/"
    "20261021T202642Z/historical_pregame_elo.csv"
)

LOCK_DIR = ROOT / (
    "reports/v58_development/elo_prospective_locks_v221/"
    "20261021T204450423907Z"
)

HIST_SHA256 = (
    "8326672e26162d0904bcdc95f7100b22e7b19ebf4a7df0e8b7a6665539d28115"
)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def parse_datetime(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timezone missing")
    return dt

def test_historical_snapshot_unchanged():
    assert HIST.is_file()
    assert sha256(HIST) == HIST_SHA256

def test_historical_game_keys_unique():
    rows = read_csv(HIST)
    assert len(rows) == 19292

    keys = [(r["season"], r["game_id"]) for r in rows]
    assert len(keys) == len(set(keys))

def test_historical_elo_eligibility():
    rows = read_csv(HIST)
    eligible = [
        r for r in rows
        if r["fbs_fbs"].lower() == "true"
        and r["both_elo"].lower() == "true"
    ]

    assert len(eligible) == 4973

    for row in eligible:
        assert row["home_pregame_elo"]
        assert row["away_pregame_elo"]

        home = float(row["home_pregame_elo"])
        away = float(row["away_pregame_elo"])
        difference = float(row["elo_diff_home"])

        assert abs((home - away) - difference) < 1e-8

def test_prospective_lock_integrity():
    manifest = json.loads(
        (LOCK_DIR / "manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )

    predictions = LOCK_DIR / "locked_predictions.csv"

    assert sha256(predictions) == (
        manifest["locked_predictions_sha256"]
    )

    rows = read_csv(predictions)

    assert len(rows) == 20
    assert len({r["game_id"] for r in rows}) == 20

    for row in rows:
        kickoff = parse_datetime(row["kickoff_utc_forecast"])
        locked = parse_datetime(row["locked_at_utc"])
        captured = parse_datetime(row["elo_capture_dt"])
        source_mtime = parse_datetime(
            row["forecast_file_mtime_utc"]
        )

        assert locked < kickoff
        assert captured < kickoff
        assert source_mtime < kickoff

        original = float(row["original_margin_home"])
        elo = float(row["elo_margin_home"])
        adjustment = float(row["elo_adjustment"])

        assert abs(original + adjustment - elo) < 1e-8

def test_prospective_locked_inputs_unchanged():
    manifest = json.loads(
        (LOCK_DIR / "manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )

    for name, expected in manifest["input_hashes"].items():
        path = ROOT / name.replace("\\", "/")
        assert path.is_file()
        assert sha256(path) == expected

def test_elo_not_certified_for_production():
    manifest = json.loads(
        (LOCK_DIR / "manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )

    assert manifest["historical_elo_pit_certified"] is False
    assert manifest["source_feature_cutoffs_certified"] is False
    assert manifest["production_model_changed"] is False
