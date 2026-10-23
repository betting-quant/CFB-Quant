"""Offline chronology gate regressions."""

import json
import pandas as pd

from cfb_quant.live.pregame_chronology_gate_v58 import check


def fixture(tmp_path, created, generated):
    bridge = tmp_path / "bridge.json"
    forecast = tmp_path / "forecast.json"
    snapshot = tmp_path / "features.parquet"

    asof = "2026-10-21T21:55:42Z"

    bridge.write_text(json.dumps({
        "asof_utc": asof,
        "created_at_utc": created,
    }), encoding="utf-8")

    forecast.write_text(json.dumps({
        "generated_at_utc": generated,
        "snapshot_metadata": {"asof_utc": asof},
        "snapshot_validation": {"snapshot_asof_utc": asof},
    }), encoding="utf-8")

    pd.DataFrame({
        "game_id": ["401871070", "401871070"],
        "kickoff_utc": ["2026-10-22T00:00:00Z"] * 2,
    }).to_parquet(snapshot, index=False)

    return bridge, forecast, snapshot


def test_post_kickoff_build_rejected(tmp_path):
    paths = fixture(
        tmp_path,
        "2026-10-22T14:17:36Z",
        "2026-10-22T14:18:53Z",
    )
    result = check(*paths)
    assert result["chronology_status"] == "REJECT"
    assert "SNAPSHOT_CREATED_AFTER_KICKOFF" in result["reasons"]
    assert "FORECAST_GENERATED_AFTER_KICKOFF" in result["reasons"]


def test_prospective_timing_not_source_certification(tmp_path):
    paths = fixture(
        tmp_path,
        "2026-10-21T22:00:00Z",
        "2026-10-21T22:01:00Z",
    )
    result = check(*paths)
    assert result["chronology_status"] == "PASS"
    assert result["lineage_status"] == "UNVERIFIED"
    assert result["production_forecast_authorized"] is False


def test_asof_after_kickoff_rejected(tmp_path):
    paths = fixture(
        tmp_path,
        "2026-10-22T00:01:00Z",
        "2026-10-22T00:02:00Z",
    )
    bridge = json.loads(paths[0].read_text(encoding="utf-8"))
    bridge["asof_utc"] = "2026-10-22T00:00:00Z"
    paths[0].write_text(json.dumps(bridge), encoding="utf-8")

    forecast = json.loads(paths[1].read_text(encoding="utf-8"))
    forecast["snapshot_metadata"]["asof_utc"] = bridge["asof_utc"]
    forecast["snapshot_validation"]["snapshot_asof_utc"] = bridge["asof_utc"]
    paths[1].write_text(json.dumps(forecast), encoding="utf-8")

    result = check(*paths)
    assert result["chronology_status"] == "REJECT"
    assert "DECLARED_ASOF_NOT_BEFORE_KICKOFF" in result["reasons"]


def test_forecast_before_creation_fails(tmp_path):
    import pytest
    paths = fixture(
        tmp_path,
        "2026-10-21T23:00:00Z",
        "2026-10-21T22:00:00Z",
    )
    with pytest.raises(ValueError, match="predates"):
        check(*paths)
