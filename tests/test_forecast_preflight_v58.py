"""Tests for research-only v5.8 forecast preflight."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from cfb_quant.live.forecast_preflight_v58 import preflight
from tests.test_data_freshness_gate_v58 import (
    CUTOFF,
    fixture_root,
    write_metadata,
)


def test_fresh_inputs_allow_research_review_only(tmp_path):
    root, *_ = fixture_root(tmp_path)
    result = preflight(root, CUTOFF.isoformat())

    assert result["data_freshness"]["status"] == "READY"
    assert result["can_proceed_to_research_review"] is True
    assert result["can_publish_production_forecast"] is False
    assert result["historical_point_in_time_certified"] is False
    assert result["source_completeness_independently_certified"] is False


def test_stale_input_blocks_research_review(tmp_path):
    root, _, _, advanced = fixture_root(tmp_path)
    write_metadata(advanced, retrieved="2026-10-16T19:47:24Z")

    result = preflight(root, CUTOFF.isoformat())

    assert result["data_freshness"]["status"] == "STALE"
    assert result["can_proceed_to_research_review"] is False
    assert result["can_publish_production_forecast"] is False


def test_future_cutoff_rejected(tmp_path):
    root, *_ = fixture_root(tmp_path)
    future = (
        datetime.now(timezone.utc) + timedelta(days=1)
    ).isoformat()

    with pytest.raises(ValueError, match="future"):
        preflight(root, future)


def test_invalid_source_age_rejected(tmp_path):
    root, *_ = fixture_root(tmp_path)

    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            preflight(root, CUTOFF.isoformat(), value)


def test_invalid_cutoff_rejected(tmp_path):
    root, *_ = fixture_root(tmp_path)

    with pytest.raises((ValueError, TypeError)):
        preflight(root, "not-a-timestamp")
