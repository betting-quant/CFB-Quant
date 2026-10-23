import hashlib
import json
from datetime import datetime, timezone

import pytest

from cfb_quant.spread_total.elo_incremental_v58 import (
    merge_fbs_captures,
)
from cfb_quant.spread_total.elo_prospective_gate_v58 import (
    prospective_elo_candidates,
)

ENDPOINT = "https://api.collegefootballdata.com/games"


def game(gid, kickoff, home, away, completed=False,
         hp=None, ap=None):
    return {
        "id": gid,
        "startDate": kickoff,
        "homeTeam": home,
        "awayTeam": away,
        "homeClassification": "fbs",
        "awayClassification": "fbs",
        "completed": completed,
        "homePoints": hp,
        "awayPoints": ap,
    }


def pack(records, observed="2026-10-23T17:30:47Z"):
    raw = json.dumps(records).encode("utf-8")
    return raw, {
        "source_endpoint": ENDPOINT,
        "retrieved_at_utc": observed,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_only_future_games_are_candidates():
    archive = pack([
        game(1, "2026-10-20T18:00:00Z",
             "A", "B", True, 28, 14),
        game(2, "2026-10-24T18:00:00Z",
             "A", "C"),
    ])
    merged = merge_fbs_captures([archive])
    output = prospective_elo_candidates(merged)

    assert output["candidate_count"] == 1
    assert output["candidates"][0].game_id == "2"
    assert output["candidates"][0].home_elo > 1500
    assert output["updates_applied"] == 1


def test_game_at_exact_capture_cutoff_excluded():
    archive = pack([
        game(1, "2026-10-23T17:30:47Z", "A", "B"),
        game(2, "2026-10-24T18:00:00Z", "C", "D"),
    ])
    output = prospective_elo_candidates(
        merge_fbs_captures([archive])
    )
    assert [s.game_id for s in output["candidates"]] == ["2"]


def test_all_past_games_yield_no_candidates():
    archive = pack([
        game(1, "2026-10-20T18:00:00Z",
             "A", "B", True, 28, 14),
    ])
    output = prospective_elo_candidates(
        merge_fbs_captures([archive])
    )
    assert output["candidate_count"] == 0


def test_no_eligible_games_rejected():
    archive = pack([{
        **game(1, "2026-10-24T18:00:00Z", "A", "B"),
        "awayClassification": "fcs",
    }])
    with pytest.raises(ValueError, match="No eligible"):
        prospective_elo_candidates(
            merge_fbs_captures([archive])
        )


def test_invalid_cutoff_rejected():
    archive = pack([
        game(1, "2026-10-24T18:00:00Z", "A", "B"),
    ])
    merged = merge_fbs_captures([archive])
    merged["latest_capture_utc"] = datetime(2026, 10, 23)

    with pytest.raises(ValueError, match="timezone-aware"):
        prospective_elo_candidates(merged)


def test_research_not_marked_production_approved():
    archive = pack([
        game(1, "2026-10-24T18:00:00Z", "A", "B"),
    ])
    output = prospective_elo_candidates(
        merge_fbs_captures([archive])
    )
    assert output["historical_pit_certified"] is False
    assert output["production_approved"] is False
