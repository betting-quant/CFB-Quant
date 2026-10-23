"""Offline pre-save chronology regressions."""

import pandas as pd
import pytest

from cfb_quant.live.pregame_presave_gate_v58 import assess


def frame():
    return pd.DataFrame({
        "game_id": ["1", "1"],
        "kickoff_utc": ["2026-10-22T00:00:00Z"] * 2,
    })


def meta(created="2026-10-21T22:00:00Z"):
    return {
        "asof_utc": "2026-10-21T21:00:00Z",
        "created_at_utc": created,
    }


def test_normal_pregame_chronology():
    result = assess(
        meta(), frame(), "2026-10-21T22:01:00Z"
    )
    assert result["status"] == "PASS"
    assert result["classification"] == "CHRONOLOGY_VALID_RESEARCH"
    assert result["prospective_performance_eligible"] is False
    assert result["source_availability_certified"] is False


def test_postkickoff_creation_rejected():
    result = assess(
        meta("2026-10-22T14:00:00Z"),
        frame(),
        "2026-10-22T14:01:00Z",
    )
    assert result["status"] == "REJECT"
    assert "FEATURE_CREATED_AFTER_KICKOFF" in result["reasons"]


def test_archive_override_remains_retrospective():
    result = assess(
        meta("2026-10-22T14:00:00Z"),
        frame(),
        "2026-10-22T14:01:00Z",
        allow_started=True,
    )
    assert result["status"] == "PASS"
    assert result["classification"] == "RETROSPECTIVE_RESEARCH"
    assert result["prospective_performance_eligible"] is False


def test_future_asof_rejected_even_for_archive():
    metadata = meta()
    metadata["asof_utc"] = "2026-10-22T00:01:00Z"
    metadata["created_at_utc"] = "2026-10-22T00:02:00Z"
    result = assess(
        metadata, frame(), "2026-10-22T00:03:00Z",
        allow_started=True,
    )
    assert result["status"] == "REJECT"
    assert "ASOF_NOT_BEFORE_KICKOFF" in result["reasons"]


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone aware"):
        assess(meta(), frame(), "2026-10-21T22:01:00")


def test_creation_after_generation_rejected():
    result = assess(
        meta("2026-10-21T23:00:00Z"),
        frame(),
        "2026-10-21T22:00:00Z",
    )
    assert result["status"] == "REJECT"
    assert "FEATURE_CREATED_AFTER_FORECAST_GENERATION" in result["reasons"]
