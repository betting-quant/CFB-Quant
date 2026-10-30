"""Regression tests for historical chronology and leakage detection."""

from pathlib import Path

import pytest

from cfb_quant.features.chronology_leakage_guard_v1 import (
    audit_future_result_leakage,
    require_clean_chronology,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / (
    "data/processed/cfb_matchup_features_20261029.parquet"
)

KICKOFF = ROOT / (
    "reports/spread_total_schedule_chronology_v2/"
    "team_game_kickoff_map.csv"
)


def test_oct29_legacy_features_fail_certification():
    if not DATASET.exists() or not KICKOFF.exists():
        pytest.skip("Local historical artifacts unavailable")

    result = audit_future_result_leakage(DATASET, KICKOFF)

    assert result["unsafe_predecessor_rows"] >= 34
    assert result["stored_lag_matches_future_result"] >= 34
    assert result["missing_kickoff_rows"] >= 230

    with pytest.raises(ValueError):
        require_clean_chronology(result)


def test_clean_contract_accepts_complete_order():
    result = {
        "missing_kickoff_rows": 0,
        "unsafe_predecessor_rows": 0,
        "stored_lag_matches_future_result": 0,
    }

    assert require_clean_chronology(result)


def test_missing_kickoff_fails_closed():
    result = {
        "missing_kickoff_rows": 1,
        "unsafe_predecessor_rows": 0,
        "stored_lag_matches_future_result": 0,
    }

    with pytest.raises(ValueError, match="Missing verified kickoff"):
        require_clean_chronology(result)
