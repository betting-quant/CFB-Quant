from datetime import datetime, timezone

import pytest

from cfb_quant.spread_total.elo_source_adapter_v58 import (
    adapt_cfbd_games,
)
from cfb_quant.spread_total.elo_replay_v58 import replay_games


def schedule(gid, start, home="A", away="B", **extras):
    return {
        "id": gid,
        "startDate": start,
        "homeTeam": home,
        "awayTeam": away,
        **extras,
    }


def capture(gid, observed, home=28, away=14, completed=True):
    return {
        "game_id": gid,
        "captured_at_utc": observed,
        "completed": completed,
        "home_points": home,
        "away_points": away,
    }


T1 = "2025-09-01T18:00:00Z"
T2 = "2025-09-01T21:00:00Z"
T3 = "2025-09-02T18:00:00Z"


def test_schedule_only_never_imports_retrospective_scores():
    record = schedule(
        1, T1,
        completed=True,
        homePoints=28,
        awayPoints=14,
    )
    game = adapt_cfbd_games([record])[0]
    assert game.home_points is None
    assert game.away_points is None
    assert game.result_available_utc is None


def test_explicit_completed_capture_creates_update():
    games = adapt_cfbd_games(
        [schedule(1, T1), schedule(2, T3, "A", "C")],
        [capture(1, T2)],
    )
    result = replay_games(games)
    assert result.snapshots[1].home_elo > 1500
    assert result.updates_applied == 1


def test_capture_after_next_kickoff_cannot_leak():
    games = adapt_cfbd_games(
        [schedule(1, T1), schedule(2, T2, "A", "C")],
        [capture(1, T3)],
    )
    result = replay_games(games)
    assert result.snapshots[1].home_elo == 1500


def test_result_capture_at_next_kickoff_not_applied():
    games = adapt_cfbd_games(
        [schedule(1, T1), schedule(2, T2, "A", "C")],
        [capture(1, T2)],
    )
    assert replay_games(games).snapshots[1].home_elo == 1500


def test_incomplete_capture_rejected():
    with pytest.raises(ValueError, match="completed"):
        adapt_cfbd_games(
            [schedule(1, T1)],
            [capture(1, T2, completed=False)],
        )


def test_missing_capture_timestamp_rejected():
    with pytest.raises(ValueError, match="captured_at_utc"):
        adapt_cfbd_games(
            [schedule(1, T1)],
            [capture(1, None)],
        )


def test_naive_capture_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        adapt_cfbd_games(
            [schedule(1, T1)],
            [capture(1, "2025-09-01T21:00:00")],
        )


def test_capture_before_kickoff_rejected():
    with pytest.raises(ValueError, match="after kickoff"):
        adapt_cfbd_games(
            [schedule(1, T2)],
            [capture(1, T1)],
        )


def test_duplicate_schedule_id_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        adapt_cfbd_games([
            schedule(1, T1),
            schedule(1, T2),
        ])


def test_duplicate_result_capture_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        adapt_cfbd_games(
            [schedule(1, T1)],
            [capture(1, T2), capture(1, T3)],
        )


def test_unknown_result_game_rejected():
    with pytest.raises(ValueError, match="corresponding"):
        adapt_cfbd_games(
            [schedule(1, T1)],
            [capture(99, T2)],
        )


def test_invalid_final_score_rejected():
    with pytest.raises(ValueError, match="home_points"):
        adapt_cfbd_games(
            [schedule(1, T1)],
            [capture(1, T2, home=-1)],
        )


def test_schedule_requires_timezone_aware_kickoff():
    with pytest.raises(ValueError, match="timezone-aware"):
        adapt_cfbd_games([
            schedule(1, "2025-09-01T18:00:00"),
        ])


def test_output_sorted_by_kickoff_not_input_order():
    output = adapt_cfbd_games([
        schedule(2, T2, "C", "D"),
        schedule(1, T1, "A", "B"),
    ])
    assert [g.game_id for g in output] == ["1", "2"]


def test_observed_result_timestamp_retained_exactly():
    games = adapt_cfbd_games(
        [schedule(1, T1)],
        [capture(1, T2)],
    )
    assert games[0].result_available_utc == datetime(
        2025, 9, 1, 21, tzinfo=timezone.utc
    )
