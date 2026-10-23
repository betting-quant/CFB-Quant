"""Regression tests for v5.8 source reconciliation."""

import hashlib
import json

import pandas as pd
import pytest

from cfb_quant.live.source_reconciliation_v58 import reconcile


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_data(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()

    schedule = [{
        "id": 123456,
        "season": 2026,
        "seasonType": "regular",
        "week": 7,
        "startDate": "2026-10-17T18:00:00Z",
        "completed": True,
        "homeTeam": "Home FBS",
        "awayTeam": "Away FBS",
        "homeId": 101,
        "awayId": 202,
        "homeClassification": "fbs",
        "awayClassification": "fbs",
        "homePoints": 28,
        "awayPoints": 21,
    }]

    schedule_path = archive / "cfbd_games.json"
    write_json(schedule_path, schedule)

    audit = {
        "fresh_games": 1,
        "request_completed_utc": "2026-10-20T19:06:12Z",
        "snapshot_sha256": hashlib.sha256(
            schedule_path.read_bytes()
        ).hexdigest(),
    }
    write_json(archive / "audit.json", audit)

    history_path = tmp_path / "history.parquet"
    history = pd.DataFrame([
        {
            "game_id": 123456,
            "home_away": "home",
            "team": "Home FBS",
            "team_id": 101,
            "points": 28,
        },
        {
            "game_id": 123456,
            "home_away": "away",
            "team": "Away FBS",
            "team_id": 202,
            "points": 21,
        },
    ])
    history.to_parquet(history_path, index=False)

    advanced_path = tmp_path / "advanced.json"
    write_json(advanced_path, [
        {"gameId": 123456, "team": "Home FBS"},
        {"gameId": 123456, "team": "Away FBS"},
    ])

    return archive, history_path, advanced_path


def test_complete_reconciliation_passes(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    result = reconcile(archive, history, advanced)

    assert result["status"] == "PASS"
    assert result["required_games"] == 1
    assert result["valid_history_pairs"] == 1
    assert result["valid_advanced_pairs"] == 1
    assert result["issue_count"] == 0
    assert result["historical_point_in_time_certified"] is False


def test_corrupted_schedule_fingerprint_rejected(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    path = archive / "cfbd_games.json"
    games = json.loads(path.read_text(encoding="utf-8"))
    games[0]["homePoints"] = 35
    write_json(path, games)

    with pytest.raises(ValueError, match="fingerprint"):
        reconcile(archive, history, advanced)


def test_missing_completed_game_fails(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    frame = pd.read_parquet(history)
    frame.iloc[:0].to_parquet(history, index=False)

    result = reconcile(archive, history, advanced)

    assert result["status"] == "FAIL"
    assert result["issue_count"] > 0
    assert any("history missing" in x for x in result["issues"])


def test_history_score_mismatch_fails(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    frame = pd.read_parquet(history)
    frame.loc[frame.home_away == "home", "points"] = 35
    frame.to_parquet(history, index=False)

    result = reconcile(archive, history, advanced)

    assert result["status"] == "FAIL"
    assert any("score mismatch" in x for x in result["issues"])


def test_wrong_advanced_team_fails(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    write_json(advanced, [
        {"gameId": 123456, "team": "Wrong Team"},
        {"gameId": 123456, "team": "Away FBS"},
    ])

    result = reconcile(archive, history, advanced)

    assert result["status"] == "FAIL"
    assert any("advanced pair mismatch" in x for x in result["issues"])


def test_extra_history_game_fails(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    frame = pd.read_parquet(history)
    extra = frame.copy()
    extra["game_id"] = 999999

    pd.concat(
        [frame, extra], ignore_index=True
    ).to_parquet(history, index=False)

    result = reconcile(archive, history, advanced)

    assert result["status"] == "FAIL"
    assert result["extra_history_games"] == 1
    assert any("outside archived" in x for x in result["issues"])


def test_missing_advanced_pair_fails(tmp_path):
    archive, history, advanced = fixture_data(tmp_path)

    write_json(advanced, [
        {"gameId": 123456, "team": "Home FBS"},
    ])

    result = reconcile(archive, history, advanced)

    assert result["status"] == "FAIL"
    assert result["valid_advanced_pairs"] == 0


def rewrite_schedule_and_manifest(archive, schedule):
    """Update synthetic archive while keeping its fingerprint valid."""
    schedule_path = archive / "cfbd_games.json"
    write_json(schedule_path, schedule)

    audit_path = archive / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["snapshot_sha256"] = hashlib.sha256(
        schedule_path.read_bytes()
    ).hexdigest()
    write_json(audit_path, audit)


def test_unknown_classification_cannot_pass(tmp_path):
    """A completed game with unknown eligibility must not receive PASS."""
    archive, history, advanced = fixture_data(tmp_path)

    schedule_path = archive / "cfbd_games.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule[0]["homeClassification"] = None
    rewrite_schedule_and_manifest(archive, schedule)

    # Without any history, the current implementation incorrectly
    # treats this game as outside the required universe.
    frame = pd.read_parquet(history)
    frame.iloc[:0].to_parquet(history, index=False)
    write_json(advanced, [])

    result = reconcile(archive, history, advanced)

    assert result["status"] != "PASS"
    assert result["issue_count"] > 0


def test_missing_completed_score_cannot_pass(tmp_path):
    """A completed FBS game with missing scores cannot be silently skipped."""
    archive, history, advanced = fixture_data(tmp_path)

    schedule_path = archive / "cfbd_games.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule[0]["homePoints"] = None
    rewrite_schedule_and_manifest(archive, schedule)

    frame = pd.read_parquet(history)
    frame.iloc[:0].to_parquet(history, index=False)
    write_json(advanced, [])

    result = reconcile(archive, history, advanced)

    assert result["status"] != "PASS"
    assert result["issue_count"] > 0


def test_unknown_fcs_opponent_without_history_fails(tmp_path):
    """Missing classification cannot justify excluding a completed game."""
    archive, history, advanced = fixture_data(tmp_path)

    schedule_path = archive / "cfbd_games.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))

    schedule[0]["homeClassification"] = None
    schedule[0]["awayClassification"] = "fcs"

    rewrite_schedule_and_manifest(archive, schedule)

    frame = pd.read_parquet(history)
    frame.iloc[:0].to_parquet(history, index=False)
    write_json(advanced, [])

    result = reconcile(archive, history, advanced)

    assert result["status"] == "FAIL"
    assert result["issue_count"] > 0
    assert any(
        "classification" in issue.lower()
        for issue in result["issues"]
    )



def test_reconciliation_rejects_post_cutoff_archive(tmp_path):
    """October 20 retrieval cannot certify an October 17 forecast."""
    archive, history, advanced = fixture_data(tmp_path)

    with pytest.raises(ValueError, match="after forecast cutoff"):
        reconcile(
            archive, history, advanced,
            forecast_cutoff_utc="2026-10-17T15:00:00Z",
        )


def test_reconciliation_accepts_earlier_retrieval(tmp_path):
    """Schedule retrieval before the proposed cutoff is permissible."""
    archive, history, advanced = fixture_data(tmp_path)

    result = reconcile(
        archive, history, advanced,
        forecast_cutoff_utc="2026-10-21T12:00:00Z",
    )

    assert result["status"] == "PASS"
    assert result["historical_point_in_time_certified"] is False


def test_reconciliation_rejects_naive_cutoff(tmp_path):
    """A forecast cutoff must have an explicit timezone."""
    archive, history, advanced = fixture_data(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        reconcile(
            archive, history, advanced,
            forecast_cutoff_utc="2026-10-21T12:00:00",
        )
