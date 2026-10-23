from datetime import datetime, timedelta, timezone

import pytest

from cfb_quant.spread_total.elo_replay_v58 import (
    ReplayGame,
    replay_games,
)


def t(hours):
    return datetime(2025, 9, 1, tzinfo=timezone.utc) + timedelta(
        hours=hours
    )


def game(gid, kickoff, home, away, hp, ap, result_time):
    return ReplayGame(
        game_id=gid,
        kickoff_utc=t(kickoff),
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        result_available_utc=t(result_time),
    )


def test_same_time_results_use_frozen_ratings():
    games = [
        game("1", 1, "A", "B", 28, 14, 5),
        game("2", 2, "A", "C", 21, 7, 5),
        ReplayGame("3", t(6), "A", "D"),
    ]

    result = replay_games(games)
    snapshots = {s.game_id: s for s in result.snapshots}

    assert snapshots["1"].home_elo == 1500
    assert snapshots["2"].home_elo == 1500

    # Both wins independently contribute +12 at K=24.
    assert snapshots["3"].home_elo == pytest.approx(1524)
    assert result.final_ratings["B"] == pytest.approx(1488)
    assert result.final_ratings["C"] == pytest.approx(1488)


def test_same_time_results_are_input_order_independent():
    games = [
        game("1", 1, "A", "B", 28, 14, 5),
        game("2", 2, "A", "C", 21, 7, 5),
        ReplayGame("3", t(6), "A", "D"),
    ]

    assert replay_games(games) == replay_games(
        list(reversed(games))
    )


def test_result_at_kickoff_remains_unavailable():
    games = [
        game("1", 1, "A", "B", 28, 14, 5),
        ReplayGame("2", t(5), "A", "C"),
    ]

    result = replay_games(games)
    assert result.snapshots[1].home_elo == 1500


def test_distinct_result_times_update_sequentially():
    games = [
        game("1", 1, "A", "B", 28, 14, 4),
        game("2", 2, "A", "C", 21, 7, 5),
        ReplayGame("3", t(6), "A", "D"),
    ]

    result = replay_games(games)
    assert result.snapshots[2].home_elo > 1500
    assert result.snapshots[2].home_elo < 1524
