import hashlib
import json

from cfb_quant.spread_total.elo_eligibility_v58 import (
    adapt_fbs_capture,
)
from cfb_quant.spread_total.elo_replay_v58 import replay_games


def make_game(gid, home, away, hc, ac, kickoff, completed=False,
              hp=None, ap=None):
    return {
        "id": gid,
        "homeTeam": home,
        "awayTeam": away,
        "homeClassification": hc,
        "awayClassification": ac,
        "startDate": kickoff,
        "completed": completed,
        "homePoints": hp,
        "awayPoints": ap,
    }


def pack(records):
    raw = json.dumps(records).encode("utf-8")
    return raw, {
        "source_endpoint": "https://api.collegefootballdata.com/games",
        "retrieved_at_utc": "2026-10-23T17:30:47Z",
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_fbs_matchups_only():
    raw, manifest = pack([
        make_game(1, "A", "B", "fbs", "fbs",
                  "2026-10-01T18:00:00Z", True, 28, 14),
        make_game(2, "A", "C", "fbs", "fcs",
                  "2026-10-02T18:00:00Z", True, 35, 7),
        make_game(3, "D", "E", "iii", "iii",
                  "2026-10-02T18:00:00Z", True, 0, 0),
        make_game(4, "A", "F", "fbs", "fbs",
                  "2026-10-24T18:00:00Z"),
    ])

    data = adapt_fbs_capture(raw, manifest)
    assert data["eligible_games"] == 2
    assert data["ineligible_games"] == 2
    assert data["eligible_completed_results"] == 1
    assert {g.game_id for g in data["games"]} == {"1", "4"}

    result = replay_games(data["games"])
    assert result.updates_applied == 1
    assert result.snapshots[-1].home_elo > 1500


def test_invalid_fbs_score_still_quarantined():
    raw, manifest = pack([
        make_game(1, "A", "B", "fbs", "fbs",
                  "2026-10-01T18:00:00Z", True, None, None),
    ])
    data = adapt_fbs_capture(raw, manifest)
    assert data["eligible_games"] == 1
    assert data["eligible_completed_results"] == 0
    assert replay_games(data["games"]).updates_applied == 0


def test_classification_missing_is_not_assumed_fbs():
    raw, manifest = pack([
        make_game(1, "A", "B", None, "fbs",
                  "2026-10-24T18:00:00Z"),
    ])
    data = adapt_fbs_capture(raw, manifest)
    assert data["eligible_games"] == 0


def test_fbs_fcs_outcome_cannot_affect_fbs_only_rating():
    raw, manifest = pack([
        make_game(1, "A", "B", "fbs", "fcs",
                  "2026-10-01T18:00:00Z", True, 70, 0),
        make_game(2, "A", "C", "fbs", "fbs",
                  "2026-10-24T18:00:00Z"),
    ])
    data = adapt_fbs_capture(raw, manifest)
    result = replay_games(data["games"])
    assert result.snapshots[0].home_elo == 1500
    assert result.updates_applied == 0
