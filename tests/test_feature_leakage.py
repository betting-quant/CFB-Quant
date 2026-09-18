import numpy as np
import pandas as pd
import pandas.testing as pdt

from cfb_quant.features.engine import build_features, FORBIDDEN_DIRECT_FEATURES


def _sample_data() -> pd.DataFrame:
    rows = []
    for week, game_id in [(1, 101), (2, 102), (3, 103)]:
        for team, opponent, home_away, id_base in [
            ("Alpha", "Beta", "home", 10),
            ("Beta", "Alpha", "away", 20),
        ]:
            rows.append(
                {
                    "game_id": game_id,
                    "season": 2025,
                    "week": week,
                    "season_type": "regular",
                    "player_id": id_base,
                    "player_name": f"{team} QB",
                    "team": team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "team_points": 20 + week,
                    "opponent_points": 17 + week,
                    "completions": 18 + week,
                    "pass_attempts": 28 + week,
                    "passing_yards": 230 + 10 * week,
                    "passing_touchdowns": 2,
                    "interceptions": 1,
                    "rush_attempts": 5 + week,
                    "rushing_yards": 20 + 2 * week,
                    "rushing_touchdowns": 0,
                    "receptions": 0,
                    "receiving_yards": 0,
                    "receiving_touchdowns": 0,
                    "qbr": 60 + week,
                    "dropbacks": 31 + week,
                    "sacks_taken": 2,
                    "pass_epa": 2.0 + week / 10,
                    "pass_successes": 15,
                    "total_air_yards": 280 + 5 * week,
                    "red_zone_dropbacks": 4,
                    "early_down_dropbacks": 20,
                    "late_down_dropbacks": 11,
                    "completion_rate_pbp": 0.64,
                    "pass_success_rate": 0.48,
                    "epa_per_dropback": 0.08,
                    "air_yards_per_attempt": 9.0,
                    "rush_plays_pbp": 6,
                    "non_kneel_carries": 5,
                    "kneels": 1,
                    "rush_epa": 0.4,
                    "rush_successes": 3,
                    "red_zone_carries": 1,
                    "short_yardage_carries": 1,
                    "power_carries": 0,
                    "explosive_rushes": 1,
                    "rush_success_rate": 0.5,
                    "rush_epa_per_play": 0.07,
                    "targets_pbp": 0,
                    "receiving_epa": 0.0,
                    "receiving_successes": 0,
                    "total_target_air_yards": 0,
                    "total_yac": 0,
                    "red_zone_targets": 0,
                    "explosive_pass_targets": 0,
                    "catch_rate_pbp": np.nan,
                    "receiving_success_rate": np.nan,
                    "receiving_epa_per_target": np.nan,
                    "air_yards_per_target": np.nan,
                }
            )
            rows.append(
                {
                    "game_id": game_id,
                    "season": 2025,
                    "week": week,
                    "season_type": "regular",
                    "player_id": id_base + 1,
                    "player_name": f"{team} WR",
                    "team": team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "team_points": 20 + week,
                    "opponent_points": 17 + week,
                    "completions": 0,
                    "pass_attempts": 0,
                    "passing_yards": 0,
                    "passing_touchdowns": 0,
                    "interceptions": 0,
                    "rush_attempts": 1,
                    "rushing_yards": 4,
                    "rushing_touchdowns": 0,
                    "receptions": 5 + week,
                    "receiving_yards": 70 + 5 * week,
                    "receiving_touchdowns": 1,
                    "qbr": np.nan,
                    "targets_pbp": 8 + week,
                    "receiving_epa": 1.2,
                    "receiving_successes": 5,
                    "total_target_air_yards": 105,
                    "total_yac": 25,
                    "red_zone_targets": 2,
                    "explosive_pass_targets": 2,
                    "catch_rate_pbp": 0.70,
                    "receiving_success_rate": 0.60,
                    "receiving_epa_per_target": 0.15,
                    "air_yards_per_target": 12.0,
                }
            )
    return pd.DataFrame(rows)


def _row(frame: pd.DataFrame, player_id: int, week: int) -> pd.Series:
    return frame.loc[(frame["player_id"] == player_id) & (frame["week"] == week)].iloc[0]


def test_forbidden_current_game_columns_not_in_feature_list():
    _, feature_cols = build_features(_sample_data())
    assert not (set(feature_cols) & FORBIDDEN_DIRECT_FEATURES)


def test_lag1_uses_previous_game_only():
    frame, _ = build_features(_sample_data())
    week2 = _row(frame, 10, 2)
    assert week2["p_pass_attempts_lag1"] == 29


def test_current_game_mutation_does_not_change_own_features():
    raw = _sample_data()
    base, feature_cols = build_features(raw)

    mutated = raw.copy()
    mask = (mutated["player_id"] == 10) & (mutated["week"] == 2)
    for col in [
        "completions",
        "pass_attempts",
        "passing_yards",
        "rush_attempts",
        "rushing_yards",
        "team_points",
        "qbr",
        "dropbacks",
        "pass_epa",
        "red_zone_dropbacks",
    ]:
        if col in mutated.columns:
            mutated.loc[mask, col] = 9999

    changed, _ = build_features(mutated)

    left = _row(base, 10, 2)[feature_cols]
    right = _row(changed, 10, 2)[feature_cols]
    pdt.assert_series_equal(left, right, check_names=False)


def test_future_game_mutation_does_not_change_past_features():
    raw = _sample_data()
    base, feature_cols = build_features(raw)

    mutated = raw.copy()
    future = mutated["week"] == 3
    for col in [
        "completions",
        "pass_attempts",
        "passing_yards",
        "rush_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
        "team_points",
        "opponent_points",
        "targets_pbp",
        "pass_epa",
        "rush_epa",
        "receiving_epa",
    ]:
        if col in mutated.columns:
            mutated.loc[future, col] = 7777

    changed, _ = build_features(mutated)

    base_past = base.loc[base["week"] <= 2, ["game_id", "player_id", *feature_cols]].reset_index(drop=True)
    changed_past = changed.loc[changed["week"] <= 2, ["game_id", "player_id", *feature_cols]].reset_index(drop=True)
    pdt.assert_frame_equal(base_past, changed_past)
