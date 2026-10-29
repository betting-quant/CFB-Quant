import pandas as pd
import pytest

from cfb_quant.features import matchup_engine as engine


def make_rows():
    rows = []

    games = [
        (200, "2026-09-05T20:00:00Z", 40.0, 10.0),
        (100, "2026-08-29T20:00:00Z", 20.0, 14.0),
        (300, "2026-09-12T20:00:00Z", 30.0, 21.0),
    ]

    for game_id, kickoff, home_points, away_points in games:
        for team, opponent, home_away, points, opp_points in [
            ("Alpha", "Beta", "home", home_points, away_points),
            ("Beta", "Alpha", "away", away_points, home_points),
        ]:
            rows.append({
                "game_id": game_id,
                "season": 2026,
                "season_type": "regular",
                "week": 1 if game_id in (100, 200) else 2,
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "points": points,
                "opponent_points": opp_points,
                "kickoff_utc": pd.Timestamp(kickoff),
            })

    return pd.DataFrame(rows)


def test_chronological_sort_uses_kickoff():
    frame = make_rows()
    ordered = engine._sort_frame(frame)

    alpha = ordered.loc[ordered.team == "Alpha"]

    assert alpha.game_id.tolist() == [100, 200, 300]


def test_offense_uses_correct_previous_game():
    frame = make_rows()
    ordered = engine._sort_frame(frame)

    result, _ = engine.add_offense_history(ordered)
    alpha = result[result.team == "Alpha"].set_index("game_id")

    assert alpha.loc[200, "off_points_lag1"] == 20.0
    assert alpha.loc[300, "off_points_lag1"] == 40.0


def test_defense_uses_correct_previous_game():
    frame = make_rows()
    ordered = engine._sort_frame(frame)

    result, _ = engine.build_defense_history(ordered)

    beta = result[
        result.defense_team == "Beta"
    ].set_index("game_id")

    assert beta.loc[200, "def_allowed_points_lag1"] == 20.0
    assert beta.loc[300, "def_allowed_points_lag1"] == 40.0


def test_missing_kickoff_rejected():
    frame = make_rows()
    frame.loc[0, "kickoff_utc"] = pd.NaT

    with pytest.raises(ValueError, match="kickoff"):
        engine._sort_frame(frame)


def test_legacy_sort_preserved():
    frame = make_rows().drop(columns="kickoff_utc")
    ordered = engine._sort_frame(frame)

    alpha = ordered[ordered.team == "Alpha"]

    assert alpha.game_id.tolist() == [100, 200, 300]


def test_six_hour_cutoff_excludes_recent_game():
    cutoff = pd.Timestamp(
        "2026-10-29T23:30:00Z"
    ) - pd.Timedelta(hours=6)

    sample = pd.DataFrame({
        "game_id": [1, 2, 3],
        "kickoff": pd.to_datetime([
            "2026-10-28T20:00:00Z",
            "2026-10-29T20:00:00Z",
            "2026-10-30T00:00:00Z",
        ], utc=True),
    })

    eligible = sample[sample.kickoff < cutoff]

    assert eligible.game_id.tolist() == [1]


def test_live_history_cutoff_excludes_later_games():
    from cfb_quant.live.matchup_live import (
        _prepare_kickoff_aware_history,
    )

    # Test the live cutoff independently of v5.8's verified
    # kickoff map, which may cover fewer games than the raw
    # local history. Production map validation stays unchanged.
    import glob

    history_files = sorted(
        glob.glob(engine.DEFAULT_INPUT_GLOB)
    )

    if not history_files:
        raise RuntimeError("No historical team-game files found.")

    history = pd.concat(
        [pd.read_parquet(f) for f in history_files],
        ignore_index=True,
    )

    # Historical simulation cutoff, not a live forecast.
    target = {
        "id": 999999999,
        "startDate": "2026-10-14T23:30:00Z",
    }

    eligible, kickoff = _prepare_kickoff_aware_history(
        history,
        [target],
        safety_hours=6,
    )

    cutoff = kickoff - pd.Timedelta(hours=6)

    assert len(eligible) > 0
    assert len(eligible) < len(history)
    assert eligible["kickoff"].notna().all()
    assert (eligible["kickoff"] < cutoff).all()

    # Verify actual 2026 records were excluded.
    excluded_ids = (
        set(history["game_id"])
        - set(eligible["game_id"])
    )
    assert excluded_ids

    print(
        "\nHistorical rows excluded:",
        len(history) - len(eligible),
    )


def test_kickoff_mode_rejects_mixed_target_times():
    from cfb_quant.live.matchup_live import (
        _prepare_kickoff_aware_history,
    )

    games = [
        {"startDate": "2026-10-29T23:30:00Z"},
        {"startDate": "2026-10-30T01:00:00Z"},
    ]

    with pytest.raises(ValueError, match="one target kickoff"):
        _prepare_kickoff_aware_history(
            pd.DataFrame(),
            games,
        )


def test_kickoff_mode_rejects_missing_target_time():
    from cfb_quant.live.matchup_live import (
        _prepare_kickoff_aware_history,
    )

    with pytest.raises(ValueError, match="kickoff timestamp"):
        _prepare_kickoff_aware_history(
            pd.DataFrame(),
            [{"startDate": None}],
        )


def test_kickoff_mode_rejects_negative_safety_hours():
    from cfb_quant.live.matchup_live import (
        _prepare_kickoff_aware_history,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        _prepare_kickoff_aware_history(
            pd.DataFrame(),
            [{"startDate": "2026-10-29T23:30:00Z"}],
            safety_hours=-1,
        )
