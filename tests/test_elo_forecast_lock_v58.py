from datetime import datetime, timezone

import pytest

from cfb_quant.spread_total.elo_forecast_lock_v58 import (
    prepare_lock,
    canonical_bytes,
    content_sha256,
)
from cfb_quant.spread_total.elo_replay_v58 import (
    ReplayGame,
)


def fake_merged():
    from cfb_quant.spread_total.elo_incremental_v58 import (
        merge_fbs_captures,
    )
    import json
    import hashlib

    records = [
        {
            "id": 1,
            "startDate": "2026-10-20T18:00:00Z",
            "homeTeam": "A",
            "awayTeam": "B",
            "homeClassification": "fbs",
            "awayClassification": "fbs",
            "completed": True,
            "homePoints": 28,
            "awayPoints": 14,
        },
        {
            "id": 2,
            "startDate": "2026-10-24T18:00:00Z",
            "homeTeam": "A",
            "awayTeam": "C",
            "homeClassification": "fbs",
            "awayClassification": "fbs",
            "completed": False,
            "homePoints": None,
            "awayPoints": None,
        },
        {
            "id": 3,
            "startDate": "2026-11-15T18:00:00Z",
            "homeTeam": "D",
            "awayTeam": "E",
            "homeClassification": "fbs",
            "awayClassification": "fbs",
            "completed": False,
            "homePoints": None,
            "awayPoints": None,
        },
    ]
    raw = json.dumps(records).encode()
    manifest = {
        "source_endpoint": "https://api.collegefootballdata.com/games",
        "retrieved_at_utc": "2026-10-23T17:30:47Z",
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return merge_fbs_captures([(raw, manifest)])


LOCK = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)


def test_only_near_term_future_locked():
    result = prepare_lock(fake_merged(), LOCK)
    assert result["forecast_count"] == 1
    assert result["forecasts"][0]["game_id"] == "2"
    assert result["rating_updates_applied"] == 1


def test_lock_timestamp_cannot_predate_capture():
    with pytest.raises(ValueError, match="before"):
        prepare_lock(
            fake_merged(),
            datetime(2026, 10, 22, tzinfo=timezone.utc),
        )


def test_naive_lock_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_lock(fake_merged(), datetime(2026, 10, 23))


def test_zero_horizon_rejected():
    with pytest.raises(ValueError, match="horizon"):
        prepare_lock(fake_merged(), LOCK, horizon_hours=0)


def test_fingerprint_is_deterministic():
    a = canonical_bytes(prepare_lock(fake_merged(), LOCK))
    b = canonical_bytes(prepare_lock(fake_merged(), LOCK))
    assert a == b
    assert content_sha256(a) == content_sha256(b)


def test_no_forecast_is_marked_production_approved():
    result = prepare_lock(fake_merged(), LOCK)
    assert result["historical_pit_certified"] is False
    assert result["production_approved"] is False
    assert result["model"]["k_factor"] == 24.0
