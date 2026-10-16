import pandas as pd
import pytest
from cfb_quant.spread_total.feature_preflight import audit_frame


def rows():
    return pd.DataFrame([
        {"game_id": 1, "season": 2024, "week": 1, "team": "A", "opponent": "B", "home_away": "home", "points": 21, "off_points_lag1": None, "def_allowed_points_lag1": 30},
        {"game_id": 1, "season": 2024, "week": 1, "team": "B", "opponent": "A", "home_away": "away", "points": 14, "off_points_lag1": 10, "def_allowed_points_lag1": 22},
        {"game_id": 2, "season": 2024, "week": 5, "team": "A", "opponent": "C", "home_away": "home", "points": 28, "off_points_lag1": 21, "def_allowed_points_lag1": 14},
    ])


def test_complete_and_incomplete_pairs_preserved():
    out = audit_frame(rows(), {"feature_columns": ["off_points_lag1", "def_allowed_points_lag1"]})
    assert out["summary"]["complete_pairs"] == 1
    assert out["summary"]["incomplete_pairs"] == 1
    assert out["summary"]["opponent_identity_mismatches_on_complete_pairs"] == 0
    assert out["period_missingness"].loc[0, "team_games"] == 2


def test_reject_current_game_points_feature():
    with pytest.raises(ValueError, match="Current-game"):
        audit_frame(rows(), {"feature_columns": ["points"]})


def test_reject_duplicate_team_game():
    f = pd.concat([rows(), rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate team-game"):
        audit_frame(f, {"feature_columns": ["off_points_lag1"]})


def test_detect_opponent_mismatch():
    f = rows()
    f.loc[1, "opponent"] = "Z"
    o = audit_frame(f, {"feature_columns": ["off_points_lag1"]})
    assert o["summary"]["opponent_identity_mismatches_on_complete_pairs"] == 1
