from datetime import datetime, timedelta, timezone

import pytest

from cfb_quant.spread_total.elo_replay_v58 import (
    ReplayGame,
    replay_games,
)


def t(hours):
    return datetime(2025, 9, 1, tzinfo=timezone.utc) + timedelta(hours=hours)


def game(gid, kickoff, home, away, hp=None, ap=None, available=None):
    return ReplayGame(
        game_id=gid,
        kickoff_utc=t(kickoff),
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        result_available_utc=t(available) if available is not None else None,
    )


def test_empty_replay():
    result = replay_games([])
    assert result.snapshots == ()
    assert result.updates_applied == 0


def test_first_game_uses_initial_ratings():
    result = replay_games([game("1", 1, "A", "B")])
    snap = result.snapshots[0]
    assert snap.home_elo == 1500
    assert snap.away_elo == 1500
    assert snap.expected_home_win == 0.5


def test_result_updates_next_game_after_availability():
    result = replay_games([
        game("1", 1, "A", "B", 28, 14, 4),
        game("2", 6, "A", "C"),
    ])
    assert result.snapshots[1].home_elo > 1500
    assert result.updates_applied == 1


def test_unavailable_result_cannot_leak():
    result = replay_games([
        game("1", 1, "A", "B", 28, 14, 8),
        game("2", 6, "A", "C"),
    ])
    assert result.snapshots[1].home_elo == 1500


def test_same_time_as_result_does_not_allow_update():
    result = replay_games([
        game("1", 1, "A", "B", 28, 14, 6),
        game("2", 6, "A", "C"),
    ])
    assert result.snapshots[1].home_elo == 1500


def test_simultaneous_games_are_independent():
    result = replay_games([
        game("2", 1, "C", "D", 20, 10, 4),
        game("1", 1, "A", "B", 30, 10, 4),
    ])
    assert all(s.home_elo == 1500 for s in result.snapshots)
    assert [s.game_id for s in result.snapshots] == ["1", "2"]


def test_order_of_input_does_not_change_output():
    games = [
        game("2", 6, "A", "C", 7, 24, 9),
        game("1", 1, "A", "B", 28, 14, 4),
    ]
    assert replay_games(games) == replay_games(list(reversed(games)))


def test_duplicate_game_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        replay_games([
            game("1", 1, "A", "B"),
            game("1", 6, "C", "D"),
        ])


def test_simultaneous_team_conflict_rejected():
    with pytest.raises(ValueError, match="simultaneous"):
        replay_games([
            game("1", 1, "A", "B"),
            game("2", 1, "A", "C"),
        ])


def test_missing_result_timestamp_rejected():
    with pytest.raises(ValueError, match="Scores require"):
        replay_games([game("1", 1, "A", "B", 10, 7)])


def test_future_score_does_not_affect_prior_snapshot():
    result = replay_games([
        game("1", 1, "A", "B", 14, 21, 8),
        game("2", 2, "A", "C", 24, 10, 5),
        game("3", 6, "A", "D"),
    ])
    assert result.snapshots[1].home_elo == 1500
    assert result.snapshots[2].home_elo > 1500


def test_invalid_result_before_kickoff_rejected():
    with pytest.raises(ValueError, match="after kickoff"):
        replay_games([game("1", 4, "A", "B", 10, 7, 2)])


def test_timezone_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone"):
        replay_games([
            ReplayGame("1", datetime(2025, 9, 1), "A", "B")
        ])
