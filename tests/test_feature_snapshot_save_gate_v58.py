"""Tests for the feature snapshot save boundary."""

import pandas as pd
import pytest

from cfb_quant.live.feature_snapshot_save_gate_v58 import assess


def frame():
    return pd.DataFrame({
        "game_id": ["100", "100"],
        "kickoff_utc": ["2026-10-22T00:00:00Z"] * 2,
    })


def metadata():
    return {
        "asof_utc": "2026-10-21T21:55:00Z",
        "no_current_game_outcomes": True,
    }


def test_valid_real_time_save():
    result = assess(
        metadata(), frame(), "2026-10-21T23:00:00Z"
    )
    assert result["status"] == "PASS"
    assert result["source_availability_certified"] is False


def test_postkickoff_save_rejected():
    result = assess(
        metadata(), frame(), "2026-10-22T14:00:00Z"
    )
    assert result["status"] == "REJECT"
    assert "ACTUAL_SAVE_AT_OR_AFTER_KICKOFF" in result["reasons"]


def test_inside_buffer_rejected():
    result = assess(
        metadata(), frame(), "2026-10-21T23:59:30Z"
    )
    assert result["status"] == "REJECT"
    assert "INSUFFICIENT_PREKICKOFF_SAVE_BUFFER" in result["reasons"]


def test_exact_buffer_allowed():
    result = assess(
        metadata(), frame(), "2026-10-21T23:59:00Z"
    )
    assert result["status"] == "PASS"


def test_future_asof_rejected():
    info = metadata()
    info["asof_utc"] = "2026-10-22T00:01:00Z"
    result = assess(
        info, frame(), "2026-10-21T23:00:00Z"
    )
    assert result["status"] == "REJECT"
    assert "ASOF_AFTER_ACTUAL_SAVE_TIME" in result["reasons"]


def test_missing_outcome_assurance_rejected():
    info = metadata()
    info["no_current_game_outcomes"] = False
    result = assess(
        info, frame(), "2026-10-21T23:00:00Z"
    )
    assert result["status"] == "REJECT"
    assert "OUTCOME_ABSENCE_NOT_CONFIRMED" in result["reasons"]


def test_naive_save_time_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        assess(metadata(), frame(), "2026-10-21T23:00:00")


def test_invalid_buffer_rejected():
    with pytest.raises(ValueError, match="Invalid minimum"):
        assess(
            metadata(),
            frame(),
            "2026-10-21T23:00:00Z",
            minimum_lead_seconds=-1,
        )
