import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from cfb_quant.live.source_capture_guard_v58 import verify


NAMES = {
    "schedule_2026": "games_2026.json",
    "advanced_2026": "advanced_2026.json",
    "team_games_2026": "team_games_2026.parquet",
}


def fixture_capture(tmp_path):
    now = datetime.now(timezone.utc) - timedelta(minutes=2)
    completed = now + timedelta(seconds=10)

    sources = {}

    for label, filename in NAMES.items():
        data = (label + "-synthetic").encode()
        (tmp_path / filename).write_bytes(data)

        sources[label] = {
            "captured_filename": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "captured_at_utc": now.isoformat(),
            "provider_publication_time_verified": False,
            "pre_capture_availability_certified": False,
        }

    manifest = {
        "schema": "cfb_v58_source_capture_v1",
        "capture_kind": "local_file_copy",
        "research_only": True,
        "capture_completed_at_utc": completed.isoformat(),
        "historical_pit_certified": False,
        "provider_publication_times_verified": False,
        "independent_external_timestamp_verified": False,
        "safe_for_forecast_asof_before_capture": False,
        "production_modified": False,
        "sources": sources,
    }

    path = tmp_path / "capture_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, completed


def test_valid_capture(tmp_path):
    _, completed = fixture_capture(tmp_path)
    result = verify(
        tmp_path,
        (completed + timedelta(seconds=1)).isoformat()
    )
    assert result["verified"] is True
    assert result["source_count"] == 3


def test_tampered_data_rejected(tmp_path):
    fixture_capture(tmp_path)
    (tmp_path / "advanced_2026.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify(tmp_path)


def test_backdated_asof_rejected(tmp_path):
    _, completed = fixture_capture(tmp_path)
    with pytest.raises(ValueError, match="precedes"):
        verify(
            tmp_path,
            (completed - timedelta(seconds=1)).isoformat()
        )


def test_missing_source_rejected(tmp_path):
    fixture_capture(tmp_path)
    (tmp_path / "games_2026.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify(tmp_path)


def test_unsupported_certification_rejected(tmp_path):
    manifest, _ = fixture_capture(tmp_path)
    manifest["historical_pit_certified"] = True
    (tmp_path / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="provenance"):
        verify(tmp_path)


def test_invalid_filename_rejected(tmp_path):
    manifest, _ = fixture_capture(tmp_path)
    manifest["sources"]["schedule_2026"]["captured_filename"] = "../bad"
    (tmp_path / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="filename"):
        verify(tmp_path)
