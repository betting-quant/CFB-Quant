import hashlib
import json

import pytest

from cfb_quant.spread_total.elo_incremental_v58 import (
    merge_fbs_captures,
)
from cfb_quant.spread_total.elo_replay_v58 import replay_games

ENDPOINT = "https://api.collegefootballdata.com/games"
T0 = "2026-10-20T18:00:00Z"
T1 = "2026-10-21T18:00:00Z"
T2 = "2026-10-22T18:00:00Z"
T3 = "2026-10-24T18:00:00Z"


def game(gid, kickoff=T0, home="A", away="B",
         completed=False, hp=None, ap=None, hc="fbs", ac="fbs"):
    return {
        "id": gid, "startDate": kickoff,
        "homeTeam": home, "awayTeam": away,
        "homeClassification": hc, "awayClassification": ac,
        "completed": completed,
        "homePoints": hp, "awayPoints": ap,
    }


def capture(records, observed):
    raw = json.dumps(records, sort_keys=True).encode()
    return raw, {
        "source_endpoint": ENDPOINT,
        "retrieved_at_utc": observed,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_first_observation_preserved_across_captures():
    a = capture([game(1, completed=True, hp=28, ap=14),
                 game(2, T3, "A", "C")], T1)
    b = capture([game(1, completed=True, hp=28, ap=14),
                 game(2, T3, "A", "C")], T2)

    result = merge_fbs_captures([b, a])
    assert result["capture_count"] == 2
    assert result["first_results"]["1"]["observed_at_utc"] == (
        "2026-10-21T18:00:00+00:00"
    )
    assert replay_games(result["games"]).snapshots[-1].home_elo > 1500


def test_revised_score_is_detected_not_substituted():
    a = capture([game(1, completed=True, hp=28, ap=14)], T1)
    b = capture([game(1, completed=True, hp=31, ap=14)], T2)

    result = merge_fbs_captures([a, b])
    assert result["first_results"]["1"]["home_points"] == 28
    assert len(result["revisions"]) == 1
    assert result["revisions"][0]["later_score"] == (31, 14)


def test_later_valid_score_after_invalid_record():
    a = capture([game(1, completed=True, hp=None, ap=None)], T1)
    b = capture([game(1, completed=True, hp=21, ap=10)], T2)

    result = merge_fbs_captures([a, b])
    assert result["accepted_results"] == 1
    assert len(result["invalid_score_observations"]) == 1
    assert result["first_results"]["1"]["observed_at_utc"] == (
        "2026-10-22T18:00:00+00:00"
    )


def test_game_identity_conflict_rejected():
    a = capture([game(1, home="A")], T1)
    b = capture([game(1, home="Different")], T2)

    with pytest.raises(ValueError, match="Conflicting"):
        merge_fbs_captures([a, b])


def test_duplicate_capture_times_rejected():
    a = capture([game(1)], T1)
    with pytest.raises(ValueError, match="duplicate"):
        merge_fbs_captures([a, a])


def test_result_cannot_leak_into_prior_kickoff():
    a = capture([game(1, completed=True, hp=28, ap=14),
                 game(2, T1, "A", "C")], T2)
    result = merge_fbs_captures([a])
    snaps = {s.game_id: s for s in replay_games(result["games"]).snapshots}
    assert snaps["2"].home_elo == 1500


def test_non_fbs_game_excluded_from_updates():
    a = capture([
        game(1, hc="fbs", ac="fcs",
             completed=True, hp=70, ap=0),
        game(2, T3, "A", "C"),
    ], T1)
    result = merge_fbs_captures([a])
    assert result["eligible_games"] == 1
    assert result["accepted_results"] == 0


def test_capture_integrity_cannot_be_bypassed():
    raw, manifest = capture([game(1)], T1)
    with pytest.raises(ValueError, match="SHA-256"):
        merge_fbs_captures([(raw + b" ", manifest)])
