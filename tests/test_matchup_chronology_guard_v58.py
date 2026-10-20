
import pandas as pd
import pytest

from cfb_quant.features.matchup_engine import (
    _sort_frame,
    _history_features,
)


def fixture():
    return pd.DataFrame({
        "game_id": [1, 2, 3],
        "season": [2025, 2025, 2025],
        "season_type": ["regular"] * 3,
        "week": [1, 2, 3],
        "team": ["A", "A", "A"],
        "opponent": ["B", "C", "D"],
        "kickoff_utc": pd.to_datetime([
            "2025-09-01T12:00:00Z",
            "2025-09-08T12:00:00Z",
            "2025-09-15T12:00:00Z",
        ]),
        "points": [10., 20., 30.],
    })


def test_valid_history_passes():
    frame = _sort_frame(fixture())
    assert len(frame) == 3


def test_duplicate_team_kickoff_rejected():
    frame = fixture()
    frame.loc[2, "kickoff_utc"] = frame.loc[
        1, "kickoff_utc"
    ]

    with pytest.raises(ValueError, match="Duplicate team kickoff"):
        _sort_frame(frame)


def test_duplicate_defensive_kickoff_rejected():
    frame = fixture()
    frame.loc[2, "opponent"] = "C"
    frame.loc[2, "kickoff_utc"] = frame.loc[
        1, "kickoff_utc"
    ]
    frame.loc[2, "team"] = "X"

    with pytest.raises(
        ValueError, match="Duplicate defensive team kickoff"
    ):
        _sort_frame(frame)


def test_missing_timestamp_rejected():
    frame = fixture()
    frame.loc[1, "kickoff_utc"] = pd.NaT

    with pytest.raises(ValueError, match="kickoff_utc"):
        _sort_frame(frame)


def test_current_game_mutation_isolation():
    frame = _sort_frame(fixture())
    original, cols = _history_features(
        frame, group_col="team",
        metrics=["points"], prefix="off_"
    )

    changed = frame.copy()
    changed.loc[1, "points"] = 9999.

    mutated, _ = _history_features(
        changed, group_col="team",
        metrics=["points"], prefix="off_"
    )

    pd.testing.assert_series_equal(
        original.loc[1, cols],
        mutated.loc[1, cols]
    )


def test_future_game_mutation_isolation():
    frame = _sort_frame(fixture())
    original, cols = _history_features(
        frame, group_col="team",
        metrics=["points"], prefix="off_"
    )

    changed = frame.copy()
    changed.loc[2, "points"] = 9999.

    mutated, _ = _history_features(
        changed, group_col="team",
        metrics=["points"], prefix="off_"
    )

    pd.testing.assert_frame_equal(
        original.loc[:1, cols],
        mutated.loc[:1, cols]
    )
