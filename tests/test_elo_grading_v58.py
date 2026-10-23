import hashlib
import json

import pytest

from cfb_quant.spread_total.elo_grading_v58 import grade_lock

ENDPOINT = "https://api.collegefootballdata.com/games"
KICKOFF = "2026-10-24T18:00:00Z"


def record(hp=None, ap=None, completed=False, home="A",
           kickoff=KICKOFF):
    return {
        "id": 1,
        "homeTeam": home,
        "awayTeam": "B",
        "startDate": kickoff,
        "completed": completed,
        "homePoints": hp,
        "awayPoints": ap,
    }


def make_lock(p=0.7):
    payload = {
        "locked_at_utc": "2026-10-23T18:00:00Z",
        "forecast_count": 1,
        "production_approved": False,
        "forecasts": [{
            "game_id": "1",
            "home": "A",
            "away": "B",
            "kickoff_utc": KICKOFF,
            "home_win_probability": p,
            "away_win_probability": 1 - p,
        }],
    }

    raw = json.dumps(payload).encode()
    manifest = {
        "forecast_payload_sha256": hashlib.sha256(raw).hexdigest(),
        "forecast_count": 1,
        "lock_created_at_utc": payload["locked_at_utc"],
    }
    return raw, json.dumps(manifest).encode()


def capture(records, observed="2026-10-25T00:00:00Z"):
    raw = json.dumps(records).encode()
    manifest = {
        "source_endpoint": ENDPOINT,
        "retrieved_at_utc": observed,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return raw, manifest


def run(p=0.7, captures=()):
    return grade_lock(*make_lock(p), captures)


def test_missing_result_remains_pending():
    output = run()
    assert output["graded_count"] == 0
    assert output["pending_count"] == 1
    assert output["mean_brier"] is None


def test_correct_winner_probability():
    output = run(captures=[
        capture([record(28, 14, True)])
    ])
    assert output["graded_count"] == 1
    assert output["mean_brier"] == pytest.approx(0.09)
    assert output["decisive_accuracy"] == 1


def test_incorrect_winner_probability():
    output = run(captures=[
        capture([record(14, 28, True)])
    ])
    assert output["mean_brier"] == pytest.approx(0.49)
    assert output["decisive_accuracy"] == 0


def test_prekickoff_result_not_graded():
    output = run(captures=[
        capture(
            [record(28, 14, True)],
            "2026-10-24T17:00:00Z",
        ),
    ])
    assert output["pending_count"] == 1


def test_conflicting_scores_quarantined():
    output = run(captures=[
        capture([record(28, 14, True)],
                "2026-10-25T00:00:00Z"),
        capture([record(31, 14, True)],
                "2026-10-26T00:00:00Z"),
    ])
    assert output["quarantined_count"] == 1
    assert output["graded_count"] == 0


def test_identity_conflict_quarantined():
    output = run(captures=[
        capture([record(28, 14, True, home="Different")]),
    ])
    assert output["quarantined_count"] == 1


def test_corrupted_capture_rejected():
    raw, manifest = capture([record(28, 14, True)])

    with pytest.raises(ValueError, match="SHA-256"):
        run(captures=[(raw + b" ", manifest)])
