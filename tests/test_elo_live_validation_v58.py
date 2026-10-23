import hashlib
import json

import pytest

from cfb_quant.spread_total.elo_live_validation_v58 import (
    adapt_live_capture,
)
from cfb_quant.spread_total.elo_replay_v58 import replay_games


ENDPOINT = "https://api.collegefootballdata.com/games"


def record(gid, start, home, away, completed=False, hp=None, ap=None):
    return {
        "id": gid,
        "startDate": start,
        "homeTeam": home,
        "awayTeam": away,
        "completed": completed,
        "homePoints": hp,
        "awayPoints": ap,
    }


def capture(records):
    raw = json.dumps(records).encode("utf-8")
    manifest = {
        "source_endpoint": ENDPOINT,
        "retrieved_at_utc": "2026-10-23T17:30:47Z",
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return raw, manifest


def test_valid_and_invalid_results_separated():
    raw, manifest = capture([
        record(1, "2026-10-01T18:00:00Z", "A", "B",
               True, 28, 14),
        record(2, "2026-10-02T18:00:00Z", "C", "D",
               True, None, None),
        record(3, "2026-10-24T18:00:00Z", "A", "C"),
    ])

    data = adapt_live_capture(raw, manifest)
    assert data["records"] == 3
    assert data["completed"] == 2
    assert data["accepted"] == 1
    assert len(data["excluded"]) == 1
    assert data["excluded"][0]["game_id"] == "2"

    replay = replay_games(data["games"])
    assert replay.updates_applied == 1
    assert replay.snapshots[-1].home_elo > 1500


def test_invalid_scores_never_enter_replay():
    raw, manifest = capture([
        record(1, "2026-10-01T18:00:00Z", "A", "B",
               True, -1, 14),
    ])

    data = adapt_live_capture(raw, manifest)
    assert data["accepted"] == 0
    assert replay_games(data["games"]).updates_applied == 0


def test_incomplete_games_do_not_generate_results():
    raw, manifest = capture([
        record(1, "2026-10-24T18:00:00Z", "A", "B",
               False, 28, 14),
    ])

    data = adapt_live_capture(raw, manifest)
    assert data["completed"] == 0
    assert data["accepted"] == 0
    assert data["games"][0].home_points is None


def test_source_integrity_failure_is_not_bypassed():
    raw, manifest = capture([
        record(1, "2026-10-01T18:00:00Z", "A", "B",
               True, 28, 14),
    ])

    altered = raw.replace(b"28", b"38")

    with pytest.raises(ValueError, match="SHA-256"):
        adapt_live_capture(altered, manifest)


def test_result_availability_is_observation_time():
    raw, manifest = capture([
        record(1, "2026-10-01T18:00:00Z", "A", "B",
               True, 28, 14),
    ])

    data = adapt_live_capture(raw, manifest)
    assert (
        data["games"][0].result_available_utc
        == data["observed_at_utc"]
    )
