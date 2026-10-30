"""Regression coverage for kickoff-ordered feature history."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_quant.features.matchup_engine import (
    _sort_frame,
    _history_features,
)

ROOT = Path(__file__).resolve().parents[1]

CORRECTED = (
    ROOT
    / "reports/research/chronology_rebuild_v1"
    / "features_kickoff_corrected.parquet"
)


def synthetic_history():
    # Game IDs intentionally disagree with chronological order.
    return pd.DataFrame({
        "game_id": [300, 100, 200],
        "season": [2026, 2026, 2026],
        "season_type": ["regular"] * 3,
        "week": [1, 1, 1],
        "team": ["Test State"] * 3,
        "opponent": ["Alpha", "Beta", "Gamma"],
        "kickoff_utc": [
            "2026-09-12T18:00:00Z",
            "2026-08-29T18:00:00Z",
            "2026-09-05T18:00:00Z",
        ],
        "points": [30.0, 10.0, 20.0],
    })


def test_kickoff_order_overrides_game_id():
    result = _sort_frame(synthetic_history())

    assert result["game_id"].tolist() == [100, 200, 300]
    assert result["points"].tolist() == [10.0, 20.0, 30.0]


def test_rolling_history_never_uses_later_result():
    sorted_frame = _sort_frame(synthetic_history())

    result, names = _history_features(
        sorted_frame,
        group_col="team",
        metrics=["points"],
        prefix="off_",
    )

    assert "off_points_lag1" in names
    assert pd.isna(result["off_points_lag1"].iloc[0])

    np.testing.assert_allclose(
        result["off_points_lag1"].iloc[1:].to_numpy(),
        [10.0, 20.0],
    )

    assert pd.isna(result["off_points_avg3"].iloc[0])

    np.testing.assert_allclose(
        result["off_points_avg3"].iloc[1:].to_numpy(),
        [10.0, 15.0],
    )


def test_corrected_artifact_point_history():
    if not CORRECTED.is_file():
        pytest.skip("Local corrected research artifact unavailable")

    df = pd.read_parquet(
        CORRECTED,
        columns=[
            "game_id", "team", "opponent",
            "kickoff_utc", "points", "opponent_points",
            "off_points_lag1",
            "def_allowed_points_lag1",
        ],
    )

    assert not df.duplicated(["game_id", "team"]).any()
    assert df["kickoff_utc"].notna().all()

    history = df.sort_values(
        ["team", "kickoff_utc", "game_id"],
        kind="mergesort",
    ).copy()

    history["expected_off"] = (
        history.groupby("team", sort=False)["points"].shift(1)
    )

    history["expected_own_def"] = (
        history.groupby("team", sort=False)[
            "opponent_points"
        ].shift(1)
    )

    opponent_history = history[
        ["game_id", "team", "expected_own_def"]
    ].rename(columns={
        "team": "opponent",
        "expected_own_def": "expected_opp_def",
    })

    joined = history.merge(
        opponent_history,
        on=["game_id", "opponent"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    assert joined["_merge"].eq("both").all()

    for actual, expected in [
        ("off_points_lag1", "expected_off"),
        ("def_allowed_points_lag1", "expected_opp_def"),
    ]:
        np.testing.assert_allclose(
            pd.to_numeric(joined[actual]).to_numpy(
                dtype=float
            ),
            pd.to_numeric(joined[expected]).to_numpy(
                dtype=float
            ),
            rtol=0,
            atol=1e-8,
            equal_nan=True,
        )
